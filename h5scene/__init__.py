"""h5scene: per-scene HDF5 writer + H5Scene file handle, unified (T, N, …) layout. sihang 2026-11-12.

T = time axis (T=1: time-constant scene, e.g. static multi-view),
N = camera axis. Sections:

    <scene>.h5
      /rgb/{t}_{n}  uint8 1-D, image BYTES (jpg q95) — one dataset per frame,
                     key derived from (t, n); T*N keys total
      /depth        uint16 (T,N,H,W) mm, 0 AND 65535 both invalid, lossless
                    gzip4 + shuffle (bytes ~3.5x smaller than raw)       [optional]
      /mask         bool   (T,N,H,W) 1 = valid pixel, 0 = padding, gzip4 [optional]
      /K            f32    (T,N,3,3) "TN33"; (3,3) input broadcasts     [optional]
      /T            f32    (T,N,4,4) "TN44"; (4,4) input broadcasts     [optional]
      /depth_conf   f16    (T,N,H,W) 0-1 (1 = full confidence), gzip4   [optional]
      derived: s.depth_mask(t, n, conf_min=…) — (H,W) bool, not stored:
                 depth not an invalid sentinel (0 AND 65535 excluded) and,
                 if depth_conf present, conf > (0 if conf_min is None, else conf_min)
      /scene_path   utf-8 string                                       [optional]
      /meta_json    utf-8 string (original meta.json verbatim)         [optional]
      /meta/is_metric bool      /meta/is_video bool
      attrs: TNHW = (T, N, H, W) int64 array

rgb input: either a (T,N,H,W,3) or (H,W,3) uint8 array (re-encoded to jpg q95 —
cv2 jpg encoding is deterministic, so verify can byte-compare) or a per-frame
image-BYTES sequence (flat list: T frames, N=1; (T,N) object array: N cameras
per frame) stored verbatim — bytes in, bytes out.

Depth input rule: float = meters, uint16 = already mm. Either way the writer
clips d > 65.535 m to 0 (invalid) — never saturated to 65535, so both 0 and
65535 mean invalid and a read-time `>= 65535 -> 0` double-safety stays valid.

depth_conf input: 0-1 floats (clipped) or 0-65535 uint16; stored fp16 0-1.

mask input: bool/uint8, 1 = valid pixel, 0 = padding — (T,N,H,W) per pixel,
(H,W) broadcast to every (t,n), or (T,N) per slot (whole frame is padding).

K/T/depth_conf input shapes: (C,H,W)/(3,3)/(4,4) broadcast to every (t,n);
(T,N,…) stored as-is.

H5Scene is a direct file handle (no in-memory data):

    s = H5Scene(path, T, N, H, W)     # create (overwrites), TNHW pinned in attrs
    s.set_rgb(frames_or_array)        # writes /rgb directly (shape validated)
    s.set_depth(d); s.set_mask(m); s.set_K(K); s.set_pose(P); s.set_depth_conf(c)
    s.set_meta(scene_path=…, meta_json=…)
    s.close()                         # (or: with H5Scene(…) as s: …)

    s = H5Scene(path)                 # open an existing file (TNHW read from attrs)
    s.shape, s.keys, s.has(name)
    s.rgb_bytes(t, n)  /  s.rgb(t, n)          # stored bytes / decoded (H,W,3)
    s.depth(t, n, meters=…) / s.K(t, n) / s.pose(t, n) / s.mask(t, n) / s.depth_conf(t, n)
    s.scene_path, s.meta_json, s.is_metric, s.is_video

write_scene_h5 / verify_scene_h5 are the one-shot functional API (the same
layout, written in one call); verify re-reads and byte-compares.
"""
import os
import os.path as osp

import cv2
import h5py
import numpy as np

_STR = h5py.string_dtype(encoding="utf-8")


def _frame_keys(T, N):
    """Derived rgb keys: one per (t, n), row-major f"{t}_{n}". No per-frame ids stored."""
    return [f"{t}_{n}" for t in range(T) for n in range(N)]


def _broadcast(a, dtype, T, N, tail):
    """(…tail) broadcast to every (t,n) or (T,N,…tail) as-is -> (T,N,…tail)."""
    a = np.asarray(a)
    if a.ndim == len(tail):
        a = np.broadcast_to(a, (T, N) + tail)
    else:
        a = a.reshape((T, N) + tail)
    try:
        a = np.broadcast_to(a, (T, N) + tail)
    except ValueError:
        raise ValueError(f"shape {a.shape} does not broadcast to ({T}, {N}, {tail})")
    return a.astype(dtype, copy=False)


def _to_mask(mask, T, N, H, W):
    """bool/uint8 mask -> (T,N,H,W) bool (1 = valid, 0 = padding).
    Input: (T,N,H,W) pixel-wise, (H,W) broadcast to every (t,n), or (T,N)
    per-frame (which slots are padding)."""
    a = np.asarray(mask)
    if a.ndim == 2 and a.shape == (H, W):
        a = a.reshape((1, 1, H, W))
    if a.ndim == 2 and a.shape == (T, N):
        a = a.astype(bool)[:, :, None, None]
        try:
            a = np.broadcast_to(a, (T, N, H, W))
        except ValueError:
            raise ValueError(f"(T,N) mask does not broadcast to ({T}, {N}, {H}, {W})")
        return a.copy()
    try:
        a = np.broadcast_to(a, (T, N, H, W))
    except ValueError:
        raise ValueError(f"mask shape {a.shape} does not broadcast to ({T}, {N}, {H}, {W})")
    return (a != 0).astype(bool).copy()


def _to_depth_conf(conf, T, N, H, W):
    """0-1 floats (or 0-65535 uint16) -> (T,N,H,W) fp16 0-1."""
    a = np.asarray(conf)
    if a.ndim == 2:
        a = a.reshape((1, 1, H, W))
    if a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    a = np.clip(a, 0.0, 1.0)
    a = np.where(np.isfinite(a), a, 0.0)
    try:
        a = np.broadcast_to(a, (T, N, H, W))
    except ValueError:
        raise ValueError(f"depth_conf shape {a.shape} does not broadcast to ({T}, {N}, {H}, {W})")
    return a.astype(np.float16).copy()


def _to_depth_mm(depth, T, N, H, W):
    """float meters (or uint16 mm) -> (T,N,H,W) uint16 mm; d > 65.535 m -> 0 (invalid)."""
    a = np.asarray(depth)
    if a.ndim == 2:
        a = a.reshape((1, 1, H, W))
    mm = a if a.dtype == np.uint16 else (a.astype(np.float32) * 1000.0)
    mm = np.rint(np.where(np.isfinite(mm), mm, 0.0))
    mm = np.where(mm > 65535, 0, mm)  # too far -> invalid 0, never saturated 65535
    try:
        mm = np.broadcast_to(mm, (T, N, H, W))
    except ValueError:
        raise ValueError(f"depth shape {mm.shape} does not broadcast to ({T}, {N}, {H}, {W})")
    return np.clip(mm, 0, 65535).astype(np.uint16).copy()


def _normalize_rgb(rgb):
    """rgb -> (T, N, H, W, frames (T*N,) image bytes).
    Input: (T,N,H,W,3)/(H,W,3) uint8 array (re-encoded to jpg q95) or a per-frame
    image-bytes sequence (flat list/tuple: T frames, N=1; (T,N) object array:
    N cameras per frame)."""
    a = np.asarray(rgb)
    if a.dtype.kind == "S":  # list of image bytes -> fixed-size |S dtype
        a = a.astype(object)
    if a.dtype == np.uint8:
        if a.ndim == 3:
            a = a.reshape((1, 1, *a.shape))
        if a.ndim != 5 or a.shape[4] != 3:
            raise ValueError(f"rgb array must be (T,N,H,W,3) or (H,W,3), got {a.shape}")
        T, N, H, W, C = a.shape
        frames = tuple(cv2.imencode(".jpg", a[t, n],
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
                       for t in range(T) for n in range(N))
        return T, N, H, W, frames
    # bytes sequence
    if a.dtype != object:
        raise ValueError(f"rgb must be a uint8 array or an image-bytes sequence, got dtype {a.dtype}")
    if a.ndim == 1:
        flat = tuple(bytes(b) for b in a)
        T, N = len(flat), 1
    elif a.ndim == 2:
        T, N = a.shape
        flat = tuple(bytes(a[t, n]) for t in range(T) for n in range(N))
    else:
        raise ValueError(f"rgb bytes sequence must be flat (T,) or (T,N), got ndim={a.ndim}")
    img0 = cv2.imdecode(np.frombuffer(flat[0], np.uint8), cv2.IMREAD_COLOR)
    if img0 is None:
        raise ValueError("first rgb frame is not a decodable image")
    H, W = img0.shape[:2]
    return T, N, H, W, flat


class H5Scene:
    """Direct h5 file handle (no in-memory data): every set_* writes straight
    into the file (shape validated against the pinned TNHW), every read comes
    straight out of the file.

    H5Scene(path, T, N, H, W) creates/overwrites; H5Scene(path) opens."""
    def __init__(self, path, t=None, n=None, h=None, w=None, is_metric=True, is_video=True):
        if t is not None:
            if any(x is None for x in (n, h, w)):
                raise ValueError("create mode needs T, N, H, W")
            self.path = str(path)
            os.makedirs(osp.dirname(osp.abspath(self.path)), exist_ok=True)
            self._h = h5py.File(self.path, "w")
            self.T, self.N, self.H, self.W = (int(t), int(n), int(h), int(w))
            self._h.attrs["TNHW"] = np.array(self.shape, dtype=np.int64)
            self._h.create_group("rgb")
            self.set_meta(is_metric=is_metric, is_video=is_video)
        else:
            self.path = str(path)
            self._h = h5py.File(self.path, "a")
            a = self._h
            if "TNHW" not in a.attrs or "meta" not in a:
                raise ValueError(f"{self.path}: not an H5Scene file (no attrs.TNHW / /meta)")
            self.T, self.N, self.H, self.W = (int(x) for x in a.attrs["TNHW"])
            keys = _frame_keys(self.T, self.N)
            if len(a["rgb"]) != len(keys) or any(k not in a["rgb"] for k in keys):
                raise ValueError('rgb group keys != derived t_n fids')

    # ---- metadata (straight from the file) ----
    @property
    def is_metric(self):
        return bool(self._h["meta"]["is_metric"][()])

    @property
    def is_video(self):
        return bool(self._h["meta"]["is_video"][()])

    # ---- lifecycle ----
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._h is not None:
            self._h.close()
            self._h = None

    @property
    def shape(self):
        return self.T, self.N, self.H, self.W

    @property
    def keys(self):
        return _frame_keys(self.T, self.N)

    # ---- validation ----
    def _check_tnhw(self, name, tnhw):
        if tuple(tnhw) != self.shape:
            raise ValueError(f"{name} shape {tuple(tnhw)} != scene TNHW {self.shape}")

    def _write(self, name, data, **kw):
        h = self._h
        if name in h:
            del h[name]
        h.create_dataset(name, data=data, **kw)

    # ---- setters (write straight into the file) ----
    def set_rgb(self, rgb):
        """(T,N,H,W,3) uint8 / (H,W,3) / image-bytes sequence; replaces /rgb."""
        t, n, h, w, frames = _normalize_rgb(rgb)
        self._check_tnhw("rgb", (t, n, h, w))
        rg = self._h.require_group("rgb")
        for k in list(rg):
            del rg[k]
        for fid, b in zip(self.keys, frames):
            rg.create_dataset(fid, data=np.frombuffer(b, np.uint8))

    def set_depth(self, depth):
        """float meters (or uint16 mm); > 65.535 m -> 0 (invalid).
        Stored lossless gzip + shuffle: uint16 mm values rarely exceed one
        high byte (depth < 65536 mm), so shuffling low/high bytes into
        separate streams gives gzip long repeated runs — ~20% smaller than
        plain gzip on a 2x2 640x640 probe (2026-11-12)."""
        d = _to_depth_mm(depth, self.T, self.N, self.H, self.W)
        self._write("depth", d, chunks=(1, 1, self.H, self.W),
                    compression="gzip", compression_opts=4, shuffle=True)

    def set_mask(self, mask):
        """bool/uint8: (T,N,H,W) pixels / (H,W) broadcast / (T,N) slots.
        Stored gzip4 (bool is a run of 0/1 — ~180x smaller than raw on a
        mostly-valid 2x2 640x640 probe). No shuffle: 1 byte per pixel, so
        the byte-permute filter has nothing to split — measured identical
        size with it on/off."""
        self._write("mask", _to_mask(mask, self.T, self.N, self.H, self.W),
                    chunks=(1, 1, self.H, self.W), compression="gzip", compression_opts=4)

    def set_K(self, K):
        """(3,3) broadcast or (T,N,3,3) float. Stored raw (no gzip):
        measured — at typical sequence lengths (<=64 frames) gzip makes K
        *bigger* (h5 overhead + gzip block header ~ 2 KB); even-identical-K
        break-even is ~106 frames. Reconsider for T > ~100 video sequences."""
        try:
            self._write("K", _broadcast(K, np.float32, self.T, self.N, (3, 3)))
        except ValueError as e:
            raise ValueError(f"K: {e}") from None

    def set_pose(self, pose):
        """(4,4) broadcast or (T,N,4,4) float c2w. Stored raw (no gzip):
        measured 16-frame T=4096->5402 B (bigger); break-even ~64 frames.
        Reconsider for long video sequences (T > ~64)."""
        try:
            self._write("T", _broadcast(pose, np.float32, self.T, self.N, (4, 4)))
        except ValueError as e:
            raise ValueError(f"pose: {e}") from None

    def set_depth_conf(self, conf):
        """0-1 floats (or 0-65535 uint16); stored fp16 0-1, gzip4
        (no shuffle: measured 3% *larger* with it — most values are 1.0)."""
        self._write("depth_conf", _to_depth_conf(conf, self.T, self.N, self.H, self.W),
                    chunks=(1, 1, self.H, self.W), compression="gzip", compression_opts=4)

    def set_meta(self, scene_path=None, meta_json=None, is_metric=None, is_video=None):
        """scene_path/meta_json: str or None (None = remove the section, unless
        it was already absent); is_metric/is_video: bool (default True on create)."""
        h = self._h
        m = h.require_group("meta")
        for name, val in (("is_metric", is_metric), ("is_video", is_video)):
            if val is None:
                if name in m:
                    val = bool(m[name][()])
                else:
                    continue
            if name in m:
                del m[name]
            m.create_dataset(name, data=np.bool_(val))
        for name, val in (("scene_path", scene_path), ("meta_json", meta_json)):
            if val is None:
                if name in h:
                    del h[name]
                continue
            self._write(name, np.array(str(val), dtype=_STR))
        return self

    # ---- section presence ----
    def has(self, name):
        h = self._h
        return {"rgb": len(h["rgb"]) > 0, "depth": "depth" in h, "mask": "mask" in h,
                "K": "K" in h, "T": "T" in h, "depth_conf": "depth_conf" in h,
                "scene_path": "scene_path" in h, "meta_json": "meta_json" in h}[name]

    # ---- metadata (straight from the file) ----
    @property
    def scene_path(self):
        return self._h["scene_path"][()].decode() if self.has("scene_path") else None

    @property
    def meta_json(self):
        return self._h["meta_json"][()].decode() if self.has("meta_json") else None

    # ---- data (frame (t, n)) ----
    def _t(self, t, n):
        if not (0 <= t < self.T and 0 <= n < self.N):
            raise IndexError(f"frame ({t},{n}) out of range (T={self.T}, N={self.N})")

    def rgb_bytes(self, t=0, n=0):
        """Raw stored image bytes for frame (t, n)."""
        self._t(t, n)
        if not self.has("rgb"):
            raise KeyError("no rgb frames")
        return self._h["rgb"][f"{t}_{n}"][:].tobytes()

    def rgb(self, t=0, n=0):
        """Frame (t, n) decoded: (H, W, 3) uint8 RGB."""
        img = cv2.imdecode(np.frombuffer(self.rgb_bytes(t, n), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"rgb frame ({t},{n}) is not decodable")
        return img

    def _frame(self, name, t, n):
        self._t(t, n)
        if not self.has(name):
            raise KeyError(f"no {name} section")
        return self._h[name][t, n][:]

    def depth(self, t=0, n=0, meters=False):
        """Frame (t, n) depth: (H, W) uint16 mm (default) or float32 meters."""
        d = self._frame("depth", t, n).astype(np.int32)
        d[d >= 65535] = 0  # sentinel -> invalid (same as the writer)
        d = d.clip(0, 65535)
        return (d / 1000.0).astype(np.float32) if meters else d.astype(np.uint16)

    def K(self, t=0, n=0):
        return self._frame("K", t, n)

    def pose(self, t=0, n=0):
        return self._frame("T", t, n)

    def mask(self, t=0, n=0):
        return self._frame("mask", t, n)

    def depth_conf(self, t=0, n=0):
        return self._frame("depth_conf", t, n).astype(np.float32)

    def depth_mask(self, t=0, n=0, conf_min=None):
        """Frame (t, n) usable-depth mask: (H, W) bool, derived on read (not stored).
        True where depth is not an invalid sentinel (0 AND 65535 both excluded),
        AND — if a depth_conf section is present — conf passes the gate:
        conf_min None -> any confidence (conf > 0); a float -> conf > conf_min."""
        d = self._frame("depth", t, n).astype(np.int32)
        m = (d != 0) & (d != 65535)
        if self.has("depth_conf"):
            c = self._frame("depth_conf", t, n)
            m &= (c > 0.0) if conf_min is None else (c > conf_min)
        return m


def write_scene_h5(out_path, rgb, depth=None, mask=None, K=None, pose=None, depth_conf=None,
                   scene_path=None, meta_json=None, is_metric=True, is_video=True):
    """One-shot: write a complete scene h5 (creates/overwrites).
    rgb: (T,N,H,W,3) uint8 array OR a per-frame image-bytes sequence
    (see _normalize_rgb); pose: (4,4) or (T,N,4,4) c2w; optional sections
    absent when the input is None. Returns (T, N, H, W)."""
    t, n, h, w, _frames = _normalize_rgb(rgb)  # derive + validate TNHW up front
    s = H5Scene(out_path, t, n, h, w, is_metric=is_metric, is_video=is_video)
    try:
        s.set_rgb(rgb)
        if depth is not None:
            s.set_depth(depth)
        if mask is not None:
            s.set_mask(mask)
        if K is not None:
            s.set_K(K)
        if pose is not None:
            s.set_pose(pose)
        if depth_conf is not None:
            s.set_depth_conf(depth_conf)
        if scene_path is not None or meta_json is not None:
            s.set_meta(scene_path=scene_path, meta_json=meta_json)
    finally:
        s.close()
    return t, n, h, w


def verify_scene_h5(out_path, rgb, depth=None, mask=None, K=None, pose=None, depth_conf=None):
    """Re-read the h5 and check every section against the inputs. rgb frames are
    compared BYTE-equal (the writer's jpg q95 encoding is deterministic, so the
    input re-encodes to exactly the stored bytes). Returns (ok, msgs)."""
    ok, msgs = True, []

    def fail(msg):
        nonlocal ok
        ok = False
        msgs.append(msg)

    T, N, H, W, frames = _normalize_rgb(rgb)
    with H5Scene(out_path) as s:
        if s.shape != (T, N, H, W):
            fail(f"stored shape {s.shape} != input {(T, N, H, W)}")
        for i in range(T * N):
            t, n = divmod(i, N)
            fid = f"{t}_{n}"
            try:
                got = s.rgb_bytes(t, n)
            except KeyError:
                fail(f"rgb {fid} missing")
                continue
            if not np.array_equal(np.frombuffer(got, np.uint8),
                                 np.frombuffer(frames[i], np.uint8)):
                fail(f"rgb {fid} bytes mismatch")
        for name, check, expected in (
            ("depth", lambda d: _to_depth_mm(d, T, N, H, W), depth),
            ("mask", lambda m: _to_mask(m, T, N, H, W), mask),
            ("K", lambda k: _broadcast(k, np.float32, T, N, (3, 3)), K),
            ("T", lambda t: _broadcast(t, np.float32, T, N, (4, 4)), pose),
            ("depth_conf", lambda c: _to_depth_conf(c, T, N, H, W), depth_conf),
        ):
            if expected is None:
                if s.has(name):
                    fail(f"{name} section present but input is None")
                continue
            if not s.has(name) or not np.array_equal(s._h[name][:], check(expected)):
                fail(f"{name} mismatch")
    return ok, msgs
