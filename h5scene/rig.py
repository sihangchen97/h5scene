"""Rig fusion: N single-camera sequences (H5Cam, shape (T, H, W)) -> one
multi-camera H5Scene file (shape (T, N, H, W)). PROTOTYPE for feasibility —
promote to a stable API after the probe tests pass.

Rules (strict checks at merge time, see tests/test_rig.py):
- every camera must have the SAME T; is_metric must match; the section set
  (depth/mask/K/T/depth_conf present) must be identical across cameras.
- ONE axis must be identical across ALL cameras (H or W); only the other
  axis may differ. The rig stores (Hmax, Wmax) = max over cameras and pads
  smaller cameras top-left aligned (rgb black, depth/depth_conf 0 = invalid,
  mask False = padding) so the real region per (t, n) is exact.
- reading exposes rgb(t, n) only (decoded (H, W, 3)) — no bytes-level reader.
  merge copies stored bytes verbatim when the camera is already (H, W); it
  re-encodes (jpg q95) only the frames of smaller, padded cameras.
- K / pose are per-camera, no HW involved: stored (T, N, 3, 3) / (T, N, 4, 4).
- output is a plain H5Scene file (TNHW attrs, t_n rgb keys) — openable by
  H5Scene directly; no rig-specific section.
"""
import os

import cv2
import h5py
import numpy as np

from .scene import H5Scene
from .rgb import _JPG_QUALITY, _encode_jpg


class H5Cam:
    """Single-camera sequence file handle: shape (T, H, W). Same layout as
    H5Scene with N collapsed: attrs.THW, /rgb/{t}, /depth (T,h,w),
    /mask (T,H,W), /K (T,3,3), /T (T,4,4), /depth_conf (T,h,w)."""

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
        """(T,) image bytes (stored verbatim; the jpg q95 policy is applied
        by merge_rig, not here)."""
        b = np.asarray(rgb, dtype=object)
        if b.ndim != 1 or b.shape[0] != self.T:
            raise ValueError(f"rgb must be (T,) = ({self.T},), got {b.shape}")
        rg = self._h.require_group("rgb")
        for k in list(rg):
            del rg[k]
        for t, x in enumerate(b):
            rg.create_dataset(f"{t}", data=np.frombuffer(bytes(x), np.uint8))

    def set_depth(self, depth):
        """(T,H,W) uint16 mm (or smaller (T,h,w); h*W==w*H required)."""
        from .convert import _to_depth_mm, _depth_hw_ok
        d = _to_depth_mm(np.asarray(depth)[:, None], self.T, 1)[:, 0]
        h, w = d.shape[1], d.shape[2]
        if not _depth_hw_ok(h, w, self.H, self.W):
            raise ValueError(f"depth ({h},{w}) incompatible with scene ({self.H},{self.W})")
        self._w("depth", d, chunks=(1, h, w), compression="gzip",
                compression_opts=4, shuffle=True)

    def set_mask(self, mask):
        from .convert import _to_mask
        m = _to_mask(np.asarray(mask)[:, None], self.T, 1, self.H, self.W)[:, 0]
        self._w("mask", m, chunks=(1, self.H, self.W), compression="gzip", compression_opts=4)

    def set_K(self, K):
        from .convert import _broadcast
        self._w("K", _broadcast(K, np.float32, self.T, 1, (3, 3))[:, 0])

    def set_pose(self, pose):
        from .convert import _broadcast
        self._w("T", _broadcast(pose, np.float32, self.T, 1, (4, 4))[:, 0])

    def set_depth_conf(self, conf):
        from .convert import _to_depth_conf, _depth_hw_ok
        c = _to_depth_conf(np.asarray(conf)[:, None], self.T, 1)[:, 0]
        if not _depth_hw_ok(c.shape[1], c.shape[2], self.H, self.W):
            raise ValueError(f"depth_conf size incompatible with scene ({self.H},{self.W})")
        self._w("depth_conf", c, chunks=(1, c.shape[1], c.shape[2]),
                 compression="gzip", compression_opts=4)

    # reads
    def _f(self, name, t):
        if not (0 <= t < self.T):
            raise IndexError(f"t={t} out of range (T={self.T})")
        if name not in self._h:
            raise KeyError(f"no {name} section")
        return self._h[name][t][:]

    def rgb(self, t=0):
        """Frame t decoded: (H, W, 3) uint8 RGB (no bytes-level reader)."""
        if not (0 <= t < self.T):
            raise IndexError(f"t={t} out of range (T={self.T})")
        if "rgb" not in self._h or f"{t}" not in self._h["rgb"]:
            raise KeyError(f"no rgb frame {t}")
        img = cv2.imdecode(self._h["rgb"][f"{t}"][:], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"rgb frame {t} not decodable")
        return img

    def depth(self, t=0):
        return self._f("depth", t)

    def mask(self, t=0):
        return self._f("mask", t)

    def K(self, t=0):
        return self._f("K", t)

    def pose(self, t=0):
        return self._f("T", t)

    def depth_conf(self, t=0):
        return self._f("depth_conf", t)


def merge_rig(cam_paths, out_path, meta=None):
    """N H5Cam files -> one multi-camera H5Scene (T, N, Hmax, Wmax).
    meta: dict passed to H5Scene.set_meta. Returns (T, N, H, W)."""
    cams = [H5Cam(p) for p in cam_paths]
    try:
        T = {c.T for c in cams}
        if len(T) != 1:
            raise ValueError(f"camera T mismatch: {sorted(T)}")
        T = T.pop()
        met = {c.is_metric for c in cams}
        if len(met) != 1:
            raise ValueError(f"camera is_metric mismatch: {sorted(met)}")
        is_metric = met.pop()
        # STRICT (rig level): one axis must be identical across ALL cameras;
        # only the other axis may differ and is taken as max + padded.
        hs = {c.H for c in cams}
        ws = {c.W for c in cams}
        if len(hs) > 1 and len(ws) > 1:
            raise ValueError(
                f"rig needs H or W identical across cameras; got H={sorted(hs)} W={sorted(ws)}")
        H = max(c.H for c in cams)
        W = max(c.W for c in cams)
        N = len(cams)
        # per-camera which sections exist (must be the same set across cameras)
        secs = [tuple(k for k in ("depth", "mask", "K", "T", "depth_conf") if k in c._h)
                for c in cams]
        if len(set(secs)) != 1:
            raise ValueError(f"camera section sets differ: {secs}")
        secs = secs[0]

        s = H5Scene(out_path, T, N, H, W, is_metric=is_metric)
        try:
            rg = s._h.require_group("rgb")
            for n, c in enumerate(cams):
                for t in range(T):
                    b = c._h["rgb"][f"{t}"][:]
                    img = cv2.imdecode(b, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError(f"cam {n} frame {t} not decodable")
                    if img.shape[:2] != (c.H, c.W):
                        raise ValueError(f"cam {n} frame {t} size {img.shape[:2]} "
                                         f"!= declared ({c.H},{c.W})")
                    if (c.H, c.W) != (H, W):  # smaller cam: pad top-left, black fill
                        canvas = np.zeros((H, W, 3), np.uint8)
                        canvas[:c.H, :c.W] = img
                        b = _encode_jpg(canvas)
                    rg.create_dataset(f"{t}_{n}", data=np.frombuffer(b, np.uint8))
            for name, fill in (("depth", 0), ("mask", False), ("K", None),
                               ("T", None), ("depth_conf", 0.0)):
                if name not in secs:
                    continue
                _D = {"depth": ((H, W), np.uint16), "mask": ((H, W), bool),
                      "K": ((3, 3), np.float32), "T": ((4, 4), np.float32),
                      "depth_conf": ((H, W), np.float16)}
                shp, dtype = _D[name]
                arr = np.zeros((T, N, *shp), dtype)
                for n, c in enumerate(cams):
                    x0 = c._f(name, 0)   # strict: every frame must match this size
                    for t in range(T):
                        x = c._f(name, t)
                        if x.shape != x0.shape:
                            raise ValueError(f"cam {n} {name} frame {t} size {x.shape} "
                                             f"!= frame 0 {x0.shape}")
                        if name in ("K", "T") or x.shape[:2] == (H, W):
                            arr[t, n] = x
                        else:  # (h, w) < (H, W): top-left align, pad `fill`
                            a = arr[t, n]
                            a[:x.shape[0], :x.shape[1]] = x
                            a[x.shape[0]:, :] = fill
                            a[:x.shape[0], x.shape[1]:] = fill
                kw = {"depth": dict(chunks=(1, 1, H, W), compression="gzip",
                                    compression_opts=4, shuffle=True),
                      "mask": dict(chunks=(1, 1, H, W), compression="gzip",
                                   compression_opts=4),
                      "depth_conf": dict(chunks=(1, 1, H, W), compression="gzip",
                                         compression_opts=4),
                      "K": {}, "T": {}}[name]
                s._write(name, arr, **kw)
            if meta:
                s.set_meta(**meta)
        finally:
            s.close()
    finally:
        for c in cams:
            c.close()
    return T, N, H, W
