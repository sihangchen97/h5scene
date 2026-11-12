"""H5Cam: single-camera sequence file — shape (T, H, W), attrs.THW.

The scene layout with the N axis collapsed: /rgb/{t} (one dataset per frame),
/depth (T, h, w), /mask (T, H, W), /K (T, 3, 3), /T (T, 4, 4),
/depth_conf (T, h, w). Inherits the file handle + generic attr/meta accessors
from H5Base; defines its OWN meta properties (dataset/scene/seq/suffix/split/
meta_json/frame_indices/tags) on top of them; and builds its set_*/read on the
shared section encode/decode helpers in _io.py (the encoding/decoding policy
lives there, once). The (h, w) of depth / depth_conf may be < (H, W) (aspect
ratio must match); reads resize back to (H, W).
"""
import os

import numpy as np

from ._io import (decode_rgb, depth_conf_to_f32, depth_mask, depth_to_meters,
                  depth_to_mm, has, is_metric, tags_dict, write_rgb_frame)
from .convert import (_broadcast, _depth_hw_ok, _to_depth_conf, _to_depth_mm,
                      _to_mask)
from .h5base import H5Base


class H5Cam(H5Base):
    """Single-camera sequence: attrs.THW = (T, H, W). create needs T, H, W
    (is_metric defaults False); open requires attrs.THW and rgb keys 0..T-1.
    set_*/read are (T, …) methods on this class, built on _io."""

    # (name -> expected type); checked via isinstance on set_attr/set_meta
    ATTR_KEYS = {"THW": np.ndarray,        # int64 (3,) = (T, H, W)
                 "is_metric": bool}
    META_KEYS = {"dataset": str,           # top-level dataset name
                 "scene": str,             # scene id
                 "seq": str,               # sequence
                 "suffix": str,            # variant marker (train/val, camera/version)
                 "split": str,             # train/val split
                 "meta_json": str,         # source metadata verbatim
                 "frame_indices": np.ndarray,  # int64 (T,) source frame numbers
                 "tags": dict}             # /meta/tags/* utf-8

    def __init__(self, path, t=None, h=None, w=None, is_metric=False):
        if t is not None:
            if any(x is None for x in (h, w)):
                raise ValueError("create mode needs T, H, W")
            super().__init__(path, create=True)
            self.T, self.H, self.W = int(t), int(h), int(w)
            self.set_attr("THW", np.array((self.T, self.H, self.W), np.int64))
            self.set_attr("is_metric", bool(is_metric))
            self._h.create_group("rgb")
        else:
            super().__init__(path)
            thw = self.attr("THW")
            if thw is None:
                self.close()
                raise ValueError(f"{self.path}: not an H5Cam file (no attrs.THW)")
            self.T, self.H, self.W = (int(x) for x in thw)
            rg = self._h.get("rgb")          # rgb is optional (empty group is ok)
            if rg is not None and len(rg) and (len(rg) != self.T or
                    any(f"{t}" not in rg for t in range(self.T))):
                self.close()
                raise ValueError("rgb keys != 0..T-1")

    @property
    def shape(self):
        return self.T, self.H, self.W

    @property
    def is_metric(self):
        return is_metric(self._h)

    def has(self, name):
        return has(self._h, name)

    # ---- meta (this class's own keys, on top of H5Base.meta) ----
    @property
    def dataset(self):
        return self.meta("dataset")

    @property
    def scene(self):
        return self.meta("scene")

    @property
    def seq(self):
        return self.meta("seq")

    @property
    def suffix(self):
        return self.meta("suffix")

    @property
    def split(self):
        return self.meta("split")

    @property
    def meta_json(self):
        return self.meta("meta_json")

    @property
    def frame_indices(self):
        return self.meta("frame_indices")

    @property
    def tags(self):
        return tags_dict(self._h)

    # ---- convenience multi-key meta writer (on top of H5Base.set_meta) ----
    def set_meta(self, dataset=None, scene=None, seq=None, suffix=None, split=None,
                 meta_json=None, frame_indices=None, tags=None, is_metric=None):
        """The 4+ location/data keys at once: str keys: value = set, None = leave as-is; frame_indices set only when given;
        is_metric -> attrs (data-side). See H5Base.META_KEYS."""
        if is_metric is not None:
            self.set_attr("is_metric", bool(is_metric))
        for name, val in (("dataset", dataset), ("scene", scene), ("seq", seq),
                          ("suffix", suffix), ("split", split),
                          ("meta_json", meta_json), ("tags", tags)):
            if val is not None:               # None = leave that key as-is
                super().set_meta(name, val)
        if frame_indices is not None:
            super().set_meta("frame_indices", frame_indices)
        return self

    # ---- writers (T semantics; _io owns the encoding policy) ----
    def set_rgb(self, rgb):
        """(T,) image bytes / a (H, W, 3) broadcast array / (T, H, W, 3) uint8
        / file paths — each frame stored as jpg q<=95 (rgb policy in rgb.py)."""
        h, w, frames = self._rgb_frames(rgb)
        if (h, w) != (self.H, self.W):
            raise ValueError(f"rgb size ({h},{w}) != scene ({self.H},{self.W})")
        rg = self._h.require_group("rgb")
        for k in list(rg):
            del rg[k]
        for t, b in enumerate(frames):
            write_rgb_frame(self._h, f"{t}", b)

    @staticmethod
    def _rgb_frames(rgb):
        """rgb -> (H, W, frames (T,) image bytes, all jpg q<=95) — the same
        per-frame policy as rgb._load_rgb_frames, for the flat (T,) /
        (T,H,W,3) / (H,W,3) single-camera inputs."""
        from .rgb import _encode_jpg, _load_rgb_frames
        a = np.asarray(rgb)
        if isinstance(rgb, (str, os.PathLike)):
            a = a.reshape(1)
        is_paths = (a.dtype.kind == "U") or (
            a.dtype == object and a.size > 0 and isinstance(a.flat[0], (str, os.PathLike)))
        if is_paths:
            flat = tuple(open(p, "rb").read() for p in (str(a[i]) for i in range(a.size)))
            return _load_rgb_frames(flat, a.size, 1)
        if a.dtype.kind == "S":
            a = a.astype(object)
        if a.dtype == np.uint8:
            if a.ndim != 4 or a.shape[3] != 3:
                raise ValueError(f"rgb array must be (T,H,W,3), got {a.shape}")
            H, W = a.shape[1], a.shape[2]
            return H, W, tuple(_encode_jpg(a[t]) for t in range(a.shape[0]))
        if a.dtype != object:
            raise ValueError(f"rgb must be a uint8 array or an image-bytes sequence, got {a.dtype}")
        if a.ndim != 1:
            raise ValueError(f"rgb bytes sequence must be flat (T,), got ndim={a.ndim}")
        return _load_rgb_frames(tuple(bytes(b) for b in a), a.size, 1)

    def _cam(self, a):
        """a (T, h, w) — or a single (h, w) frame when T == 1 — to
        (T, 1, h, w) for the (T, N) section helpers (N=1 here)."""
        a = np.asarray(a)
        if a.ndim == 2:
            if self.T != 1:
                raise ValueError(f"got a single (h, w) frame but T={self.T} != 1")
            return a[None, None]                     # (1, 1, h, w)
        if a.ndim != 3:
            raise ValueError(f"section input must be (T, h, w) or (h, w), got {a.shape}")
        return a[:, None]                            # (T, 1, h, w)

    def set_depth(self, depth):
        """(T, H, W) float meters / uint16 mm (> 65.535 m -> 0); may be stored
        smaller (T, h, w) if the aspect ratio H/W matches. gzip4 + shuffle."""
        d = _to_depth_mm(self._cam(depth), self.T, 1)[:, 0]               # (T, h, w)
        h, w = d.shape[1], d.shape[2]
        self._check_depth_hw("depth", h, w)
        self.write_section("depth", d, chunks=(1, h, w), compression="gzip",
                          compression_opts=4, shuffle=True)

    def set_depth_conf(self, conf):
        """(T, H, W) 0-1 floats (or 0-65535 uint16); same (h, w) as depth;
        f16 0-1, gzip4 (no shuffle: most values 1.0)."""
        c = _to_depth_conf(self._cam(conf), self.T, 1)[:, 0]
        self._check_depth_hw("depth_conf", c.shape[1], c.shape[2])
        self.write_section("depth_conf", c, chunks=(1, c.shape[1], c.shape[2]),
                          compression="gzip", compression_opts=4)

    def set_mask(self, mask):
        """bool/uint8 (T, H, W) pixels / (H, W) broadcast / (T,) slots;
        full (H, W), gzip4."""
        m = _to_mask(self._cam(mask), self.T, 1, self.H, self.W)[:, 0]
        self.write_section("mask", m, chunks=(1, self.H, self.W), compression="gzip", compression_opts=4)

    def set_K(self, K):
        """(3,3) broadcast or (T, 3, 3) float; stored raw (no gzip — measured
        bigger with gzip for short sequences, break-even ~106 frames)."""
        self.write_section("K", _broadcast(K, np.float32, self.T, 1, (3, 3))[:, 0])

    def set_pose(self, pose):
        """(4,4) broadcast or (T, 4, 4) float c2w; stored raw (no gzip)."""
        self.write_section("T", _broadcast(pose, np.float32, self.T, 1, (4, 4))[:, 0])

    # ---- reads (T semantics; _io owns the decoding policy) ----
    def rgb(self, t=0):
        """Frame t decoded: (H, W, 3) uint8 RGB (no bytes-level reader)."""
        self._t(t)
        if not self.has("rgb"):
            raise KeyError("no rgb frames")
        return decode_rgb(self._h["rgb"][f"{t}"][:])

    def depth(self, t=0, meters=False):
        """Frame t depth: (H, W) uint16 mm (default) or float32 meters; a
        (h, w) < (H, W) section is resized back (nearest, sentinels re-clipped)."""
        f = self._f("depth", t)
        return depth_to_meters(f, self.H, self.W) if meters else depth_to_mm(f, self.H, self.W)

    def depth_conf(self, t=0):
        """Frame t confidence: (H, W) f32 0-1 (resized back if stored smaller)."""
        return depth_conf_to_f32(self._f("depth_conf", t), self.H, self.W)

    def mask(self, t=0):
        return self._f("mask", t)

    def K(self, t=0):
        return self._f("K", t)

    def pose(self, t=0):
        return self._f("T", t)

    def depth_mask(self, t=0, conf_min=None):
        """Frame t usable-depth mask: (H, W) bool, derived on read (not stored)
        — depth not an invalid sentinel (0 AND 65535 excluded) and, if
        depth_conf present, conf > (0 if conf_min is None, else conf_min)."""
        dm = depth_to_mm(self._f("depth", t), self.H, self.W)
        conf = depth_conf_to_f32(self._f("depth_conf", t), self.H, self.W) if self.has("depth_conf") else None
        return depth_mask(dm, conf, conf_min)

    # ---- internals ----
    def _check_depth_hw(self, name, h, w):
        if not _depth_hw_ok(h, w, self.H, self.W):
            raise ValueError(f"{name} size ({h},{w}) incompatible with scene "
                             f"({self.H},{self.W}): smaller but aspect ratio H/W must match")
        other = {"depth": "depth_conf", "depth_conf": "depth"}[name]
        if other in self._h:
            s = self._h[other].shape
            if s[1:] != (h, w):
                raise ValueError(f"{name} size ({h},{w}) != {other} size ({s[1]},{s[2]})")

    def _t(self, t):
        if not (0 <= t < self.T):
            raise IndexError(f"t={t} out of range (T={self.T})")

    def _f(self, name, t):
        self._t(t)
        if not self.has(name):
            raise KeyError(f"no {name} section")
        return self._h[name][t][:]
