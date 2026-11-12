"""H5Cam: single-camera sequence file — shape (T, H, W), attrs.THW.

Layout: /rgb/{t} (one dataset per frame, image bytes), /depth (T, h, w)
uint16 mm (gzip4), /mask (T, H, W) bool, /K (T, 3, 3) /T (T, 4, 4)
float32, /depth_conf (T, h, w) float16 — the H5Base layout with the N axis
collapsed. Reads expose rgb(t) (decoded (H, W, 3)) and the per-frame
sections; the stored jpg is an implementation detail, no bytes-level access.
"""
import os

import cv2
import h5py
import numpy as np

from .convert import _broadcast, _to_depth_conf, _to_depth_mm, _to_mask, _depth_hw_ok


class H5Cam:
    """Single-camera sequence file: shape (T, H, W). attrs.THW, /rgb/{t},
    /depth (T, h, w), /mask (T, H, W), /K (T, 3, 3), /T (T, 4, 4),
    /depth_conf (T, h, w) — the H5Base layout with the N axis collapsed."""

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
