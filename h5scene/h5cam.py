"""H5Cam: single-camera sequence file — shape (T, H, W), attrs.THW.

The scene layout with the N axis collapsed: /img/{t} (one dataset per frame),
/depth (T, h, w), /mask (T, H, W), /K (T, 3, 3), /T (T, 4, 4),
/depth_conf (T, h, w). Inherits the file handle + generic attr/meta accessors
from H5Base; declares its OWN ATTR_KEYS / META_KEYS registries; and builds its
set_*/read on the codecs/ (one code path per section — the layout table in
codecs/sections.py drives the writers and the filter policy). The (h, w) of
depth / depth_conf may be < (H, W) (aspect ratio must match); reads resize
back to (H, W).
"""
import numpy as np

from .codecs import (FRAME_SHAPE, SECTIONS, broadcast_matrix, conf_frame, depth_hw_ok,
                     depth_mask as _dm, filter_args, has,
                     is_metric as _is_metric, mm_frame, meters_frame,
                     normalize_rgb, to_conf, to_mask, to_mm, to_meters_mm)
from .codecs.rgb import decode_rgb
from .h5base import H5Base, MetaSpec

class H5Cam(H5Base):
    """Single-camera sequence: attrs.THW = (T, H, W). create needs T, H, W
    (is_metric defaults False; a constructor failure removes the half-made
    file); open requires attrs.THW and, when present, img keys 0..T-1 plus
    consistent section shapes. set_*/read are (T, …) methods on this class."""

    ATTR_KEYS = {"THW": np.ndarray,        # int64 (3,) = (T, H, W)
                 "is_metric": bool}
    META_KEYS = {"dataset": MetaSpec(str),        # top-level dataset name
                 "scene": MetaSpec(str),          # scene id
                 "seq": MetaSpec(str),            # sequence
                 "suffix": MetaSpec(str),         # variant marker (train/val, camera/version)
                 "split": MetaSpec(str),          # train/val split
                 "src_frame_idxs": MetaSpec(np.ndarray, dtype=np.int64, length="T"),
                 "tags": MetaSpec(dict)}          # /meta/tags/* utf-8 (str keys and values)

    def __init__(self, path, t=None, h=None, w=None, is_metric=False, mode=None):
        if t is not None:
            if isinstance(t, str):                 # H5Cam(p, "r+") — a mode in t's slot
                raise TypeError(f"t (frame count) got a str {t!r}; to open an "
                                f"existing file use H5Cam.open({path!r}, {t!r})")
            if any(x is None for x in (h, w)) or any(v < 1 for v in (t, h, w)):
                raise ValueError("create mode needs T, H, W >= 1")
            super().__init__(path, create=True)
            try:
                self.T, self.H, self.W = int(t), int(h), int(w)
                self.set_attr("THW", np.array((self.T, self.H, self.W), np.int64))
                self.set_attr("is_metric", bool(is_metric))
                self._h.create_group("img")
            except Exception:
                self._abort_create()
                raise
        else:
            super().__init__(path, mode=mode)
            try:
                thw = self.attr("THW")
                if thw is None:
                    raise ValueError(f"{self.path}: not an H5Cam file (no attrs.THW)")
                self.T, self.H, self.W = (int(x) for x in thw)
                rg = self._h.get("img")
                if rg is not None and len(rg) and (
                        len(rg) != self.T or any(f"{t}" not in rg for t in range(self.T))):
                    raise ValueError("img keys != 0..T-1")
                self._check_sections()
            except Exception:                   # corrupt file, bad THW, ...
                self.close()
                raise

    @classmethod
    def create(cls, path, t, h, w, is_metric=False):
        """Create a new file (overwriting); T, H, W >= 1."""
        return cls(path, t, h, w, is_metric=is_metric)

    @classmethod
    def open(cls, path, mode="r"):
        """Open an existing file ("r" or "r+")."""
        return cls(path, mode=mode)

    # ---- open-time consistency (#21) ----
    def _check_sections(self):
        for name in ("depth", "mask", "depth_conf", "K", "T"):
            if name not in self._h:
                continue
            s = self._h[name].shape
            if s[0] != self.T:
                raise ValueError(f"{name} axis 0 = {s[0]} != T = {self.T}")
            if name in ("K", "T"):
                if s[1:] != FRAME_SHAPE[name]:
                    raise ValueError(f"{name} frame shape {s[1:]} != {FRAME_SHAPE[name]}")
            elif name == "mask":
                if s[1:] != (self.H, self.W):
                    raise ValueError(f"mask shape {s[1:]} != scene ({self.H}, {self.W})")
            else:                                # depth / depth_conf: aspect ratio
                h_, w_ = s[1:]
                if not depth_hw_ok(h_, w_, self.H, self.W):
                    raise ValueError(f"{name} size {(h_, w_)} incompatible with scene "
                                     f"({self.H}, {self.W}): aspect ratio H/W must match")
                other = "depth_conf" if name == "depth" else "depth"
                if other in self._h and self._h[other].shape[1:] != (h_, w_):
                    raise ValueError(f"{name} size {(h_, w_)} != {other} size "
                                     f"{self._h[other].shape[1:]}")

    @property
    def shape(self):
        return self.T, self.H, self.W

    def __len__(self):
        return self.T

    @property
    def sections(self):
        """The present sections ('img' when any frames are stored)."""
        return {n for n in ("img", *SECTIONS) if self.has(n)}

    def __repr__(self):
        return (f"<{type(self).__name__} {self.path} shape={self.shape} "
                f"sections={sorted(self.sections)}>")

    @property
    def is_metric(self):
        return _is_metric(self._h)

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
    def src_frame_idxs(self):
        return self.meta("src_frame_idxs")

    @property
    def tags(self):
        return self.meta("tags")

    # ---- src_frame_idxs semantics: dense (== arange(T)) is not stored ----
    def _meta_value(self, name, value):
        if name == "src_frame_idxs" and (value == np.arange(self.T)).all():
            return None                             # dense: drop it
        return value

    # ---- writers (T semantics; codecs/ owns the encoding policy) ----
    def set_rgb(self, rgb):
        """(T,H,W,3) uint8 RGB array / an (H,W,3) array (T=1) / a flat (T,)
        image-bytes sequence / file paths; each frame stored as jpg q<=95
        (the policy in codecs.rgb). The frame COUNT must equal T."""
        a = np.asarray(rgb)
        if a.dtype == np.uint8 and a.ndim == 4:           # (T, H, W, 3) -> (T, 1, H, W, 3)
            rgb = a[:, None]
        h, w, frames = normalize_rgb(rgb, self.T, 1)
        if len(frames) != self.T:
            raise ValueError(f"rgb gives {len(frames)} frames but T={self.T}")
        if (h, w) != (self.H, self.W):
            raise ValueError(f"rgb size ({h},{w}) != scene ({self.H},{self.W})")
        # NOTE: replace is non-atomic (old frames deleted, then new ones
        # written): a mid-loop failure leaves a partial img group. (OCR note; a
        # temp-name+rename swap would be a larger change than this call needs.)
        rg = self._h.require_group("img")
        for k in list(rg):
            del rg[k]
        for t, b in enumerate(frames):
            rg.create_dataset(f"{t}", data=np.frombuffer(b, np.uint8))

    def set_bgr(self, bgr):
        """(T,H,W,3) uint8 BGR array / an (H,W,3) array (T=1); ARRAYS ONLY
        (image bytes / file paths are rejected — the colour order on disk is
        unknowable, use set_rgb). Channels reversed to RGB, then stored
        through the set_rgb policy (jpg q<=95)."""
        a = np.asarray(bgr)
        if a.dtype != np.uint8:
            raise TypeError(f"set_bgr wants a uint8 BGR array, got {a.dtype.name}")
        if a.ndim == 3:
            if a.shape[2] != 3:
                raise ValueError(f"set_bgr (H,W,3) needs 3 channels, got {a.shape}")
            self.set_rgb(np.ascontiguousarray(a[:, :, ::-1]))
        elif a.ndim == 4:
            if a.shape[3] != 3:
                raise ValueError(f"set_bgr (T,H,W,3) needs 3 channels, got {a.shape}")
            self.set_rgb(np.ascontiguousarray(a[:, :, :, ::-1]))
        else:
            raise ValueError(f"set_bgr must be a (T,H,W,3) or (H,W,3) uint8 BGR "
                             f"array, got {a.shape}")

    def set_depth_mm(self, depth_mm):
        """(T, H, W) INTEGER mm / an (H, W) frame (T=1); MM ONLY (a float
        means meters — use set_depth_meters). May be stored smaller (T, h, w)
        if the aspect ratio H/W matches. gzip4 + shuffle (codecs.sections)."""
        d = self._cam(depth_mm)
        d = to_mm(d, self.T)
        h, w = d.shape[1], d.shape[2]
        self._check_depth_hw("depth", h, w)
        self.write_section("depth", d, chunks=(1, h, w), **filter_args("depth"))

    def set_depth_meters(self, depth_m):
        """(T, H, W) float METERS / an (H, W) frame (T=1); METERS ONLY (an
        integer means mm — use set_depth_mm). d >= 65.535 m -> 0 (the invalid
        sentinel). Same (h, w) rules as set_depth_mm."""
        d = self._cam(depth_m)
        d = to_meters_mm(d, self.T)
        h, w = d.shape[1], d.shape[2]
        self._check_depth_hw("depth", h, w)
        self.write_section("depth", d, chunks=(1, h, w), **filter_args("depth"))

    def set_depth_conf_01(self, conf):
        """(T, H, W) float 0-1 / an (H, W) frame (T=1); 0-1 FLOATS ONLY
        (integers are rejected — they are never a 0-1 confidence). Same (h, w)
        as depth; f16 0-1, gzip4 (no shuffle: most values 1.0)."""
        c = to_conf(self._cam(conf), self.T)
        self._check_depth_hw("depth_conf", c.shape[1], c.shape[2])
        self.write_section("depth_conf", c, chunks=(1, c.shape[1], c.shape[2]),
                          **filter_args("depth_conf"))

    def set_mask(self, mask):
        """bool/uint8 (T, H, W) per pixel / (H, W) broadcast / (T,) per-frame
        slots (whole-frame padding); full (H, W), gzip4."""
        m = to_mask(self._cam_mask(mask), self.T, self.H, self.W)
        self.write_section("mask", m, chunks=(1, self.H, self.W), **filter_args("mask"))

    def set_K(self, K):
        """(3,3) broadcast or (T, 3, 3) float; stored raw (no gzip — measured
        bigger with gzip for short sequences, break-even ~106 frames)."""
        self.write_section("K", broadcast_matrix(K, self.T, (3, 3)))

    def set_pose_c2w(self, pose_c2w):
        """(4,4) broadcast or (T, 4, 4) float camera-to-world (c2w) poses;
        stored raw (no gzip)."""
        self.write_section("T", broadcast_matrix(pose_c2w, self.T, (4, 4)))

    # ---- reads (T semantics; codecs/ owns the decoding policy) ----
    def rgb(self, t=0):
        """Frame t decoded: (H, W, 3) uint8 RGB (no bytes-level reader)."""
        self._t(t)
        if not self.has("img"):
            raise KeyError("no img frames")
        return decode_rgb(self._h["img"][f"{t}"][:].tobytes())

    def bgr(self, t=0):
        """Frame t decoded as (H, W, 3) uint8 BGR (a channel-reversed copy of
        rgb(t) — for direct cv2 use)."""
        return self.rgb(t)[:, :, ::-1].copy()

    def depth_mm(self, t=0):
        """Frame t depth in MM: (H, W) uint16 (a (h, w) < (H, W) section is
        resized back — nearest, sentinels re-clipped)."""
        return mm_frame(self._f("depth", t), self.H, self.W)

    def depth_meters(self, t=0):
        """Frame t depth in METERS: (H, W) float32 (resized back like
        depth_mm)."""
        return meters_frame(self._f("depth", t), self.H, self.W)

    def depth_conf(self, t=0):
        """Frame t confidence: (H, W) f32 0-1 (resized back if stored smaller)."""
        return conf_frame(self._f("depth_conf", t), self.H, self.W)

    def mask(self, t=0):
        return self._f("mask", t)

    def K(self, t=0):
        return self._f("K", t)

    def pose_c2w(self, t=0):
        """Frame t c2w pose: (4, 4) float32 camera-to-world."""
        return self._f("T", t)

    def depth_mask(self, t=0, conf_min=None):
        """Frame t usable-depth mask: (H, W) bool, derived on read (not stored)
        — depth not an invalid sentinel (0 AND 65535 excluded) and, if
        depth_conf present, conf > (0 if conf_min is None, else conf_min)."""
        mm = mm_frame(self._f("depth", t), self.H, self.W)
        conf = conf_frame(self._f("depth_conf", t), self.H, self.W) if self.has("depth_conf") else None
        return _dm(mm, conf, conf_min)

    # ---- internals ----
    def _cam(self, a):
        """a (T, h, w) — or a single (h, w) frame when T == 1 — to (T, h, w)."""
        a = np.asarray(a)
        if a.ndim == 2:
            if self.T != 1:
                raise ValueError(f"got a single (h, w) frame but T={self.T} != 1")
            return a[None]
        if a.ndim != 3 or a.shape[0] != self.T:
            raise ValueError(f"section input must be (T=({self.T}, h, w)) or (h, w), "
                             f"got {a.shape}")
        return a

    def _cam_mask(self, a):
        """a (T, H, W) / (H, W) / (T,) mask — (T,)/(H,W) only legal when
        broadcastable (T=1 for (H,W) is fine; (T,) always)."""
        a = np.asarray(a)
        if a.ndim == 2:
            if a.shape != (self.H, self.W):
                raise ValueError(f"2-D mask must be (H, W) = ({self.H}, {self.W}), got {a.shape}")
            return a[None] if self.T == 1 else a
        if a.ndim == 1:
            if a.shape != (self.T,):
                raise ValueError(f"(T,) mask must be length T = {self.T}, got {a.shape}")
            return a
        return self._cam(a)

    def del_section(self, name):
        """Delete a stored section (/img: all its frames; /depth, /mask, …: the
        whole dataset). depth_conf must be deleted first (it must share depth's
        (h, w), so it is checked against the section being removed)."""
        if name not in SECTIONS and name != "img":
            raise ValueError(f"unknown section {name!r}; known: ("
                             f"'img', {', '.join(repr(n) for n in SECTIONS)})")
        h = self._h
        if name == "img":
            rg = h.get("img")
            if rg is None or len(rg) == 0:
                return self
            del h["img"]
        else:
            if name == "depth" and "depth_conf" in h:
                s, s2 = h["depth"].shape, h["depth_conf"].shape
                if s[1:] == s2[1:]:
                    raise ValueError("delete 'depth_conf' first (it must share "
                                     "depth's (h, w); del_section('depth_conf') "
                                     "then del_section('depth'))")
            if name not in h:
                return self
            del h[name]
        return self

    def _check_depth_hw(self, name, h, w):
        if not depth_hw_ok(h, w, self.H, self.W):
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
