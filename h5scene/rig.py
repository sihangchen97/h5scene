"""H5Cam (single-camera sequence, shape (T, H, W)) and H5CamRig (N H5Cam
-> one multi-camera H5Scene file (T, N, Hmax, Wmax), maintained via HDF5
VIRTUAL DATASETS).

H5CamRig build (verified 2026-11-13):
- STRICT checks: same T, same is_metric, same section set (depth/mask/K/T/
  depth_conf), rgb per-frame size == declared (H, W), depth/depth_conf/mask
  per-frame size == frame 0, and H or W identical across ALL cameras
  (the other axis is taken as max; both axes differ -> rejected).
- /rgb: physically copied (jpg variable-length byte strings cannot be
  virtual datasets) — verbatim for full-size cameras, decode+pad+re-encode
  (jpg q95) for smaller ones, top-left aligned, black fill.
- /depth /mask /K /T /depth_conf: VIRTUAL datasets referencing the camera
  files — 0 bytes on disk in the rig file, the cameras' compressed data
  mapped directly. Full-size camera: one block per camera; smaller camera:
  per-frame blocks (H, W) -> (t, n, :H, :W) with fillvalue covering the
  pad region (0 = invalid / False = padding) — matches the stored sentinels.
- attrs.cam_files: the source camera files. H5CamRig(open) verifies they
  all exist, because a missing source makes VDS reads SILENTLY return the
  fillvalue (0) instead of erroring.
- the result is a plain H5Scene file (attrs.TNHW, /rgb t_n keys, /meta):
  open with H5Scene; H5CamRig(open) adds the source-file guard.
"""
import os

import cv2
import h5py
import numpy as np

from .convert import _to_depth_conf, _to_depth_mm, _to_mask, _depth_hw_ok
from .rgb import _encode_jpg, _JPG_QUALITY
from .scene import H5Scene

_SECTIONS = (("depth", np.uint16, 0, (2,)),          # name, dtype, fillvalue, (h,w) rank
             ("mask", np.bool_, False, (2,)),
             ("K", np.float32, 0.0, (3, 3)),
             ("T", np.float32, 0.0, (4, 4)),
             ("depth_conf", np.float16, 0.0, (2,)))


class H5Cam:
    """Single-camera sequence file: shape (T, H, W). attrs.THW, /rgb/{t},
    /depth (T, h, w), /mask (T, H, W), /K (T, 3, 3), /T (T, 4, 4),
    /depth_conf (T, h, w) — the H5Scene layout with the N axis collapsed."""

    def __init__(self, path, t=None, h=None, w=None, is_metric=True):
        if t is not None:
            if any(x is None for x in (h, w)):
                raise ValueError("create mode needs T, H, W")
            self.path = str(path)
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._h = h5py.File(self.path, "w")
            self.T, self.H, self.W = int(t), int(h), int(w)
            self._h.attrs["THW"] = np.array((self.T, self.H, self.W), np.int64)
            self._h.attrs["is_metric"] = bool(is_metric)
            self._h.create_group("rgb")
        else:
            self.path = str(path)
            self._h = h5py.File(self.path, "r")
            if "THW" not in self._h.attrs:
                self.close()
                raise ValueError(f"{self.path}: not an H5Cam file (no attrs.THW)")
            self.T, self.H, self.W = (int(x) for x in self._h.attrs["THW"])
            if len(self._h["rgb"]) != self.T or any(
                    f"{t}" not in self._h["rgb"] for t in range(self.T)):
                self.close()
                raise ValueError("rgb keys != 0..T-1")

    @property
    def shape(self):
        return self.T, self.H, self.W

    @property
    def is_metric(self):
        return bool(self._h.attrs["is_metric"])

    def close(self):
        if self._h is not None:
            self._h.close()
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _w(self, name, data, **kw):
        h = self._h
        if name in h:
            del h[name]
        h.create_dataset(name, data=data, **kw)

    def set_rgb(self, rgb):
        """(T,) image bytes (stored verbatim; jpg q95 policy applied by
        H5CamRig, not here)."""
        b = np.asarray(rgb, dtype=object)
        if b.ndim != 1 or b.shape[0] != self.T:
            raise ValueError(f"rgb must be (T,) = ({self.T},), got {b.shape}")
        rg = self._h.require_group("rgb")
        for k in list(rg):
            del rg[k]
        for t, x in enumerate(b):
            rg.create_dataset(f"{t}", data=np.frombuffer(bytes(x), np.uint8))

    def set_depth(self, depth):
        """(T, H, W) float meters / uint16 mm (or smaller (T, h, w) with
        h*W == w*H)."""
        d = _to_depth_mm(np.asarray(depth)[:, None], self.T, 1)[:, 0]
        h, w = d.shape[1], d.shape[2]
        if not _depth_hw_ok(h, w, self.H, self.W):
            raise ValueError(f"depth ({h},{w}) incompatible with scene ({self.H},{self.W})")
        self._w("depth", d, chunks=(1, h, w), compression="gzip",
                compression_opts=4, shuffle=True)

    def set_mask(self, mask):
        m = _to_mask(np.asarray(mask)[:, None], self.T, 1, self.H, self.W)[:, 0]
        self._w("mask", m, chunks=(1, self.H, self.W), compression="gzip", compression_opts=4)

    def set_K(self, K):
        from .convert import _broadcast
        self._w("K", _broadcast(K, np.float32, self.T, 1, (3, 3))[:, 0])

    def set_pose(self, pose):
        from .convert import _broadcast
        self._w("T", _broadcast(pose, np.float32, self.T, 1, (4, 4))[:, 0])

    def set_depth_conf(self, conf):
        c = _to_depth_conf(np.asarray(conf)[:, None], self.T, 1)[:, 0]
        if not _depth_hw_ok(c.shape[1], c.shape[2], self.H, self.W):
            raise ValueError(f"depth_conf size incompatible with scene ({self.H},{self.W})")
        self._w("depth_conf", c, chunks=(1, c.shape[1], c.shape[2]),
                 compression="gzip", compression_opts=4)

    # reads (no bytes-level access; the stored jpg is an implementation detail)
    def rgb(self, t=0):
        """Frame t decoded: (H, W, 3) uint8 RGB."""
        if not (0 <= t < self.T):
            raise IndexError(f"t={t} out of range (T={self.T})")
        if "rgb" not in self._h or f"{t}" not in self._h["rgb"]:
            raise KeyError(f"no rgb frame {t}")
        img = cv2.imdecode(self._h["rgb"][f"{t}"][:], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"rgb frame {t} not decodable")
        return img

    def depth(self, t=0):
        self._frame("depth", t)

    def mask(self, t=0):
        self._frame("mask", t)

    def K(self, t=0):
        self._frame("K", t)

    def pose(self, t=0):
        self._frame("T", t)

    def depth_conf(self, t=0):
        self._frame("depth_conf", t)

    def _frame(self, name, t):
        if not (0 <= t < self.T):
            raise IndexError(f"t={t} out of range (T={self.T})")
        if name not in self._h:
            raise KeyError(f"no {name} section")
        return self._h[name][t][:]


class H5CamRig:
    """N H5Cam files -> one multi-camera file (T, N, Hmax, Wmax): /rgb
    physically copied (jpg variable-length, not virtualizable), the
    fixed-shape sections /depth /mask /K /T /depth_conf as HDF5 VIRTUAL
    datasets referencing the camera files (0 bytes on disk; compressed
    source data mapped directly; pad regions covered by fillvalue).

    Build:    H5CamRig(out_path, cam_paths, meta={...})
    Open:     H5CamRig(out_path)   # verifies attrs.cam_files exist, then H5CamRig(out_path).scene
    """

    def __init__(self, out_path, cam_paths=None, meta=None):
        self.path = str(out_path)
        if cam_paths is None:            # open
            self._h = h5py.File(self.path, "r")
            if "TNHW" not in self._h.attrs or "cam_files" not in self._h.attrs:
                self.close()
                raise ValueError(f"{self.path}: not an H5CamRig file (no attrs.TNHW / cam_files)")
            missing = [p for p in self._h.attrs["cam_files"] if not os.path.exists(p)]
            if missing:
                self.close()
                raise FileNotFoundError(
                    f"{self.path}: missing camera source file(s) — VDS would "
                    f"silently read fillvalues: {missing}")
            self.scene = H5Scene(self.path)
            self._h.close()
            self._h = self.scene._h
            return
        self._build(cam_paths, meta)

    def close(self):
        if self._h is not None:
            if hasattr(self, "scene"):
                self.scene.close()
            else:
                self._h.close()
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- build ----
    def _build(self, cam_paths, meta):
        cams = [H5Cam(p) for p in cam_paths]
        try:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
            tsets = {c.T for c in cams}
            if len(tsets) != 1:
                raise ValueError(f"camera T mismatch: {sorted(tsets)}")
            T = tsets.pop()
            mets = {c.is_metric for c in cams}
            if len(mets) != 1:
                raise ValueError(f"camera is_metric mismatch: {sorted(mets)}")
            is_metric = mets.pop()
            hs = {c.H for c in cams}
            ws = {c.W for c in cams}
            if len(hs) > 1 and len(ws) > 1:
                raise ValueError(f"rig needs H or W identical across cameras; "
                                 f"got H={sorted(hs)} W={sorted(ws)}")
            H, W, N = max(c.H for c in cams), max(c.W for c in cams), len(cams)
            secs = [tuple(k for k, *_ in _SECTIONS if k in c._h) for c in cams]
            if len(set(secs)) != 1:
                raise ValueError(f"camera section sets differ: {secs}")
            secs = secs[0]
            # strict per-frame size checks
            for n, c in enumerate(cams):
                for t in range(T):
                    img = cv2.imdecode(c._h["rgb"][f"{t}"][:], cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError(f"cam {n} rgb frame {t} not decodable")
                    if img.shape[:2] != (c.H, c.W):
                        raise ValueError(f"cam {n} rgb frame {t} size {img.shape[:2]} "
                                         f"!= declared ({c.H},{c.W})")
                for name, *_ in _SECTIONS:
                    if name in secs and name not in ("K", "T"):
                        s0 = c._h[name][0].shape
                        for t in range(1, T):
                            if c._h[name][t].shape != s0:
                                raise ValueError(f"cam {n} {name} frame {t} shape {c._h[name][t].shape} "
                                                 f"!= frame 0 {s0}")

            with h5py.File(self.path, "w", libver="latest") as h:
                h.attrs["TNHW"] = np.array((T, N, H, W), np.int64)
                h.attrs["is_metric"] = is_metric
                h.attrs["cam_files"] = np.array([os.path.abspath(c.path) for c in cams], dtype=_STR)
                # /rgb: physical copy (variable-length jpg, not virtualizable)
                rg = h.create_group("rgb")
                for n, c in enumerate(cams):
                    full = (c.H, c.W) == (H, W)
                    for t in range(T):
                        b = c._h["rgb"][f"{t}"][:]
                        if full:
                            rg.create_dataset(f"{t}_{n}", data=b)      # verbatim (b is a uint8 array)
                        else:
                            img = cv2.imdecode(b, cv2.IMREAD_COLOR)
                            canvas = np.zeros((H, W, 3), np.uint8)     # pad top-left, black
                            canvas[:c.H, :c.W] = img
                            b = _encode_jpg(canvas, _JPG_QUALITY)
                            rg.create_dataset(f"{t}_{n}", data=np.frombuffer(bytes(b), np.uint8))
                # fixed-shape sections: virtual datasets referencing the cameras
                for name, dtype, fill, rank in _SECTIONS:
                    if name not in secs:
                        continue
                    shp = (H, W) if rank == (2,) else rank
                    lay = h5py.VirtualLayout(shape=(T, N, *shp), dtype=dtype)
                    for n, c in enumerate(cams):
                        if name in ("K", "T"):        # (3,3)/(4,4) frames: one block per camera
                            lay[:, n, ...] = h5py.VirtualSource(
                                os.path.abspath(c.path), name, shape=(T, *c._h[name][0].shape))
                        elif c._h[name][0].shape == (H, W) and (c.H, c.W) == (H, W):
                            lay[:, n, ...] = h5py.VirtualSource(        # full-size frame: one block
                                os.path.abspath(c.path), name, shape=(T, H, W))
                        else:  # per-frame (h, w) blocks; pad region -> fillvalue
                            for t in range(T):
                                h_, w_ = c._h[name][t].shape
                                lay[t, n, :h_, :w_] = h5py.VirtualSource(
                                    os.path.abspath(c.path), name, shape=(h_, w_))
                    h.create_virtual_dataset(name, lay, fillvalue=fill)
                m = h.require_group("meta")       # H5Scene requires /meta on open
                for k, v in (meta or {}).items():
                    if v is not None:
                        m.create_dataset(k, data=np.array(str(v), dtype=_STR))
        finally:
            for c in cams:
                c.close()
        self._h = h5py.File(self.path, "r")
        self.scene = H5Scene(self.path)
        self._h.close()
        self._h = self.scene._h

    # ---- passthrough reads: rig.shape / rig.rgb(t, n) / rig.depth(...) / rig.has / rig.tags …
    # (everything on H5Scene, incl. the .shape property) ----
    def __getattr__(self, name):
        if name.startswith("_") or name in ("path", "scene"):
            raise AttributeError(name)
        try:
            sc = object.__getattribute__(self, "scene")
        except AttributeError:
            raise AttributeError(name)
        return getattr(sc, name)


_STR = h5py.string_dtype()
