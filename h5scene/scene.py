"""H5Scene: a direct per-scene HDF5 file handle (no in-memory data) over the
unified (T, N, H, W) layout — every set_* writes straight into the file, every
read comes straight out of the file."""
import os
import os.path as osp

import cv2
import h5py
import numpy as np

from .convert import (_broadcast, _depth_hw_ok, _frame_keys, _resize_hw,
                      _to_depth_conf, _to_depth_mm, _to_mask)
from .rgb import _normalize_rgb

_STR = h5py.string_dtype(encoding="utf-8")


class H5Scene:
    """Direct h5 file handle (no in-memory data): every set_* writes straight
    into the file (shape validated against the pinned TNHW), every read comes
    straight out of the file.

    H5Scene(path, T, N, H, W) creates/overwrites; H5Scene(path) opens.
    is_metric (attrs) defaults True; the temporal-vs-multi-view distinction is
    the T/N axes themselves (T>1 video sequence), so there is no is_video key."""
    def __init__(self, path, t=None, n=None, h=None, w=None, is_metric=True):
        if t is not None:
            if any(x is None for x in (n, h, w)):
                raise ValueError("create mode needs T, N, H, W")
            self.path = str(path)
            os.makedirs(osp.dirname(osp.abspath(self.path)), exist_ok=True)
            self._h = h5py.File(self.path, "w")
            self.T, self.N, self.H, self.W = (int(t), int(n), int(h), int(w))
            self._h.attrs["TNHW"] = np.array(self.shape, dtype=np.int64)
            self._h.create_group("rgb")
            self.set_meta(is_metric=is_metric)
        else:
            self.path = str(path)
            # 'r' (not 'a'): a missing path raises FileNotFoundError instead of
            # creating an empty orphan .h5 and leaking an open handle.
            self._h = h5py.File(self.path, "r")
            a = self._h
            if "TNHW" not in a.attrs or "meta" not in a:
                self.close()
                raise ValueError(f"{self.path}: not an H5Scene file (no attrs.TNHW / /meta)")
            self.T, self.N, self.H, self.W = (int(x) for x in a.attrs["TNHW"])
            keys = _frame_keys(self.T, self.N)
            if len(a["rgb"]) != len(keys) or any(k not in a["rgb"] for k in keys):
                self.close()
                raise ValueError('rgb group keys != derived t_n fids')

    # ---- metadata (straight from the file) ----
    @property
    def is_metric(self):
        """bool. Stored in attrs (data-side); legacy files read /meta/is_metric."""
        a = self._h
        if "is_metric" in a.attrs:
            return bool(a.attrs["is_metric"])
        return bool(a["meta"]["is_metric"][()])

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

    def _check_depth_hw(self, name, h, w):
        """(h, w) may be smaller than the scene (H, W) — the aspect ratio H/W
        must match (h*W == w*H). depth and depth_conf, when both present, must
        be stored at the same (h, w); reads resize both back to (H, W)."""
        H, W = self.H, self.W
        if not _depth_hw_ok(h, w, H, W):
            raise ValueError(
                f"{name} size ({h}, {w}) incompatible with scene (H, W) = ({H}, {W}): "
                "smaller but aspect ratio H/W must match")
        other = {"depth": "depth_conf", "depth_conf": "depth"}[name]
        if self.has(other):
            d = self._h[other].shape
            if (d[2], d[3]) != (h, w):
                raise ValueError(f"{name} size ({h}, {w}) != {other} size ({d[2]}, {d[3]})")

    def _write(self, name, data, **kw):
        h = self._h
        if name in h:
            del h[name]
        h.create_dataset(name, data=data, **kw)

    # ---- setters (write straight into the file) ----
    def set_rgb(self, rgb):
        """(T,N,H,W,3) uint8 / (H,W,3) / image-bytes sequence / paths;
        replaces /rgb (encoded to jpg q<=95 per rgb._normalize_rgb)."""
        t, n, h, w, frames = _normalize_rgb(rgb)
        self._check_tnhw("rgb", (t, n, h, w))
        rg = self._h.require_group("rgb")
        for k in list(rg):
            del rg[k]
        for fid, b in zip(self.keys, frames):
            rg.create_dataset(fid, data=np.frombuffer(b, np.uint8))

    def set_depth(self, depth):
        """float meters (or uint16 mm); > 65.535 m -> 0 (invalid).
        May be stored SMALLER than the scene (H, W): the aspect ratio H/W must
        match (h*W == w*H) and, when a depth_conf section exists, the same
        (h, w). depth(t, n) resizes the stored (h, w) back to (H, W).
        Stored lossless gzip4 + shuffle (see 2026-11-12 measurements)."""
        d = _to_depth_mm(depth, self.T, self.N)
        self._check_depth_hw("depth", d.shape[2], d.shape[3])
        self._write("depth", d, chunks=(1, 1, d.shape[2], d.shape[3]),
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
        (no shuffle: measured 3% *larger* with it — most values are 1.0).
        Must be stored at the same (h, w) as a present depth section;
        depth_conf(t, n) resizes it back to (H, W)."""
        c = _to_depth_conf(conf, self.T, self.N)
        self._check_depth_hw("depth_conf", c.shape[2], c.shape[3])
        self._write("depth_conf", c, chunks=(1, 1, c.shape[2], c.shape[3]),
                    compression="gzip", compression_opts=4)

    def set_meta(self, dataset=None, scene=None, seq=None, suffix=None, split=None,
                 meta_json=None, frame_indices=None, tags=None, is_metric=None):
        """Write the dataset-side /meta keys and the data-side attrs.is_metric
        (see docs/meta_key_decisions.md, 2026-11-13).

        str keys (dataset/scene/seq/suffix/split/meta_json): value = upsert,
        None = remove if present. dataset/scene/seq/suffix are the 4 location
        keys: (dataset, scene[, seq]) shared by several h5 = the same scene's
        multi-view cameras (N>1 multi-camera datasets); suffix marks variants
        (train/val, camera/version) — see docs/meta_key_decisions.md.
        is_metric (bool) -> attrs (data-side, depth/scale semantics).
        frame_indices: (T,) frame numbers in the source; stored only when
        NOT equal to arange(T) (dense sequences: t already is the source
        frame), None = leave as-is.
        tags: dict[str, str] merged into /meta/tags/ (existing keys replaced),
        None = leave as-is."""
        h = self._h
        m = h.require_group("meta")
        if is_metric is not None:
            if "is_metric" in m:
                del m["is_metric"]      # legacy location; reads still fall back
            h.attrs["is_metric"] = bool(is_metric)
        for name, val in (("dataset", dataset), ("scene", scene), ("seq", seq),
                          ("suffix", suffix), ("split", split), ("meta_json", meta_json)):
            if name in m:
                del m[name]
            if val is not None:
                m.create_dataset(name, data=np.array(str(val), dtype=_STR))
        if frame_indices is not None:
            fi = np.asarray(frame_indices)
            if fi.ndim != 1 or fi.shape[0] != self.T:
                raise ValueError(f"frame_indices must be (T,) = ({self.T},), got {fi.shape}")
            if "frame_indices" in m:
                del m["frame_indices"]
            if not (fi == np.arange(self.T)).all():
                m.create_dataset("frame_indices", data=fi.astype(np.int64))
        if tags is not None:
            tg = m.require_group("tags")
            for k, v in tags.items():
                if k in tg:
                    del tg[k]
                tg.create_dataset(k, data=np.array(str(v), dtype=_STR))
        return self

    # ---- section presence ----
    def has(self, name):
        h = self._h
        m = h["meta"]
        return {"rgb": len(h["rgb"]) > 0, "depth": "depth" in h, "mask": "mask" in h,
                "K": "K" in h, "T": "T" in h, "depth_conf": "depth_conf" in h,
                "is_metric": "is_metric" in h.attrs or "is_metric" in m,
                "dataset": "dataset" in m, "scene": "scene" in m,
                "seq": "seq" in m, "suffix": "suffix" in m,
                "split": "split" in m,
                "meta_json": "meta_json" in m or "meta_json" in h,
                "frame_indices": "frame_indices" in m,
                "tags": "tags" in m}[name]

    # ---- metadata (straight from the file) ----
    def _meta_str(self, name):
        """/meta/<name> as str, or None; legacy root-level <name> (old layout)."""
        h = self._h
        if name in h["meta"]:
            return h["meta"][name][()].decode()
        if name in h and isinstance(h[name], h5py.Dataset):
            return h[name][()].decode()
        return None

    @property
    def dataset(self):
        return self._meta_str("dataset")

    @property
    def scene(self):
        return self._meta_str("scene")

    @property
    def seq(self):
        return self._meta_str("seq")

    @property
    def suffix(self):
        return self._meta_str("suffix")

    @property
    def split(self):
        return self._meta_str("split")

    @property
    def meta_json(self):
        return self._meta_str("meta_json")

    @property
    def frame_indices(self):
        """(T,) int64 source frame numbers, or None (dense: t is the source frame)."""
        m = self._h["meta"]
        return m["frame_indices"][:] if "frame_indices" in m else None

    @property
    def tags(self):
        """dict[str, str] from /meta/tags/, or None when absent."""
        m = self._h["meta"]
        if "tags" not in m:
            return None
        tg = m["tags"]
        return {k: tg[k][()].decode() for k in tg}

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
        """Frame (t, n) depth: (H, W) uint16 mm (default) or float32 meters.
        A depth stored at (h, w) < (H, W) is resized back to (H, W) (nearest,
        so invalid sentinels are never interpolated)."""
        d = _resize_hw(self._frame("depth", t, n), self.H, self.W,
                       cv2.INTER_NEAREST).astype(np.int32)
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
        """Frame (t, n) confidence: (H, W) f32 0-1; a (h, w) < (H, W)
        section is resized back (nearest)."""
        return _resize_hw(self._frame("depth_conf", t, n), self.H, self.W,
                          cv2.INTER_NEAREST).astype(np.float32)

    def depth_mask(self, t=0, n=0, conf_min=None):
        """Frame (t, n) usable-depth mask: (H, W) bool, derived on read (not stored).
        True where depth is not an invalid sentinel (0 AND 65535 both excluded),
        AND — if a depth_conf section is present — conf passes the gate:
        conf_min None -> any confidence (conf > 0); a float -> conf > conf_min."""
        d = _resize_hw(self._frame("depth", t, n), self.H, self.W,
                       cv2.INTER_NEAREST).astype(np.int32)
        m = (d != 0) & (d != 65535)
        if self.has("depth_conf"):
            c = _resize_hw(self._frame("depth_conf", t, n), self.H, self.W,
                           cv2.INTER_NEAREST)
            m &= (c > 0.0) if conf_min is None else (c > conf_min)
        return m
