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
import operator
import os
import posixpath

import h5py
import numpy as np

from .codecs import (FRAME_SHAPE, SECTIONS, broadcast_matrix, check_K, check_pose,
                     conf_frame, depth_hw_ok, depth_mask as _dm, filter_args, has,
                     is_metric as _is_metric, mm_frame, meters_frame,
                     normalize_img_array, normalize_img_bytes, normalize_img_paths,
                     to_conf, to_mask, to_mm, to_meters_mm)
from .codecs.rgb import decode_rgb
from .h5base import H5Base, MetaSpec

class H5Cam(H5Base):
    """Single-camera sequence: attrs.THW = (T, H, W). create needs T, H, W
    (is_metric defaults False; a constructor failure removes the half-made
    file); open requires attrs.THW and, when present, img keys 0..T-1 plus
    consistent section shapes. set_*/read are (T, …) methods on this class."""

    ATTR_KEYS = {"file_type": str,          # always "h5cam" (pinned at create)
                  "THW": np.ndarray,        # int64 (3,) = (T, H, W)
                  "is_metric": bool}
    META_KEYS = {"dataset": MetaSpec(str),        # top-level dataset name
                  "scene": MetaSpec(str),          # scene id
                  "seq": MetaSpec(str),            # sequence
                  "suffix": MetaSpec(str),         # variant marker (train/val, camera/version)
                  "split": MetaSpec(str),          # train/val split
                  "src_frame_idxs": MetaSpec(np.ndarray, dtype=np.int64, length="T"),
                  "src_frame_ids": MetaSpec(str, dtype=np.str_, length="T"),
                                            # STRING ids (the unit is in the name)
                  "timestamps": MetaSpec(float, dtype=np.float64, length="T"),
                  "tags": MetaSpec(dict)}          # /meta/tags/* utf-8 (str keys and values)

    def __init__(self, path, mode="r"):
        """Open an existing H5Cam file ("r" or "r+"). Construction goes
        through H5Cam.create — not this."""
        super().__init__(path, mode=mode)
        try:
            ft = self._h.attrs.get("file_type")
            if ft is not None and str(ft) != "h5cam":
                raise ValueError(f"{self.path}: not an H5Cam file "
                                 f"(attrs.file_type = {ft!r})")
            thw = self.attr("THW")
            if thw is None or "is_metric" not in self._h.attrs:
                raise ValueError(f"{self.path}: not an H5Cam file "
                                 "(no attrs.THW / is_metric)")
            self.T, self.H, self.W = (int(x) for x in thw)
            self._fid_index = None              # the src_frame_ids lookup cache
            self._allocated, self._written = None, set()   # the streaming state
            rg = self._h.get("img")
            if rg is not None and len(rg) and (
                    len(rg) != self.T or any(f"{t}" not in rg for t in range(self.T))):
                raise ValueError("img keys != 0..T-1")
            self._check_sections()
        except Exception:                       # corrupt file, bad THW, ...
            self.close()
            raise

    @classmethod
    def create(cls, path, t, h, w, is_metric=False):
        """Create a new file (overwriting); T, H, W INTEGRAL and >= 1
        (2.7 raises — int() would silently truncate it; True is rejected too)."""
        if any(isinstance(v, bool) for v in (t, h, w)):
            raise TypeError(f"create needs integral T, H, W, not a bool, got "
                             f"({t!r}, {h!r}, {w!r})")
        try:
            t, h, w = operator.index(t), operator.index(h), operator.index(w)
        except TypeError:
            raise TypeError(f"create needs integral T, H, W, got "
                             f"({t!r}, {h!r}, {w!r})") from None
        if min(t, h, w) < 1:
            raise ValueError(f"create needs T, H, W >= 1, got ({t}, {h}, {w})")
        cam = object.__new__(cls)               # bypass the open-only __init__
        H5Base.__init__(cam, path, create=True)
        try:
            cam.T, cam.H, cam.W = t, h, w
            cam._fid_index = None              # the src_frame_ids lookup cache
            cam._allocated, cam._written = None, set()   # the streaming state
            cam.set_attr("file_type", "h5cam")
            cam.set_attr("THW", np.array((cam.T, cam.H, cam.W), np.int64))
            cam.set_attr("is_metric", bool(is_metric))
            cam._h.create_group("img")
        except Exception:
            cam._abort_create()
            raise
        return cam

    @classmethod
    def open(cls, path, mode="r"):
        """Open an existing file ("r" or "r+")."""
        return cls(path, mode=mode)

    @classmethod
    def peek(cls, path):
        """A dict for scene LISTING without the full open validation: file_type,
        THW, is_metric, sections, dataset, scene, seq, split, src_frame_ids.
        A file that is not an H5Cam (or is missing) raises ValueError /
        FileNotFoundError."""
        import os.path as osp
        if not osp.exists(path):
            raise FileNotFoundError(f"no such file: {path}")
        with h5py.File(str(path), "r") as h:
            if str(h.attrs.get("file_type", "")) != "h5cam":
                raise ValueError(f"{path}: not an H5Cam file "
                                 f"(attrs.file_type = {h.attrs.get('file_type')!r})")
            thw = h.attrs.get("THW")
            if thw is None:
                raise ValueError(f"{path}: not an H5Cam file (no attrs.THW)")
            m = h.get("meta")
            out = {"file_type": "h5cam",
                   "THW": tuple(int(x) for x in thw),
                   "is_metric": bool(h.attrs.get("is_metric")),
                   "sections": sorted(n for n in ("img", *SECTIONS) if n in h),
                   "dataset": None, "scene": None, "seq": None, "split": None,
                   "src_frame_ids": None}
            for k in ("dataset", "scene", "seq", "split"):
                v = m.get(k) if m is not None else None
                out[k] = v[()].decode() if v is not None else None
            v = m.get("src_frame_ids") if m is not None else None
            if v is not None:
                out["src_frame_ids"] = [x.decode() for x in v[()]]
            return out

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
    def src_frame_ids(self):
        """List[str] of the frames' source ids (/meta/src_frame_ids), or None
        when unset. STRING ids only (int-like: use src_frame_idxs)."""
        v = self.meta("src_frame_ids")
        if v is None:
            return None
        if isinstance(v, (bytes, str)):
            return [v.decode() if isinstance(v, bytes) else v]
        return [x.decode() if isinstance(x, bytes) else str(x) for x in v]

    @property
    def timestamps(self):
        """np.ndarray float64 (T,) of per-frame timestamps, or None."""
        return self.meta("timestamps")

    @property
    def tags(self):
        return self.meta("tags")

    # ---- src_frame_idxs semantics: dense (== arange(T)) is not stored ----
    def _meta_value(self, name, value):
        if name == "src_frame_idxs" and (value == np.arange(self.T)).all():
            return None                             # dense: drop it
        return value

    # ---- src_frame_ids (E1): string frame ids + O(1) lookup ----
    def set_src_frame_ids(self, ids):
        """List[str] of the frames' source ids — STRING ids ONLY (the unit is
        in the name; an int-like array is rejected, use src_frame_idxs),
        length T, UNIQUE (a duplicate is a ValueError). Row order = time
        order. Can coexist with src_frame_idxs."""
        ids = list(ids)
        for i, x in enumerate(ids):
            if not isinstance(x, str):
                raise TypeError(f"src_frame_ids must be strings (int-like: use "
                                f"src_frame_idxs); element {i} is a {type(x).__name__}")
        if len(ids) != self.T:
            raise ValueError(f"src_frame_ids must be length T = {self.T}, "
                             f"got {len(ids)}")
        dup = sorted({x for x in ids if ids.count(x) > 1})
        if dup:
            raise ValueError(f"src_frame_ids must be unique; duplicates: {dup}")
        return self.set_meta("src_frame_ids", ids)

    def index_of(self, fid):
        """The t of a source id: O(1) after the first call (the index is
        cached). KeyError on an unknown fid; ValueError when no ids are set."""
        if self._fid_index is None:
            ids = self.src_frame_ids
            if ids is None:
                raise ValueError("no src_frame_ids stored (set_src_frame_ids)")
            self._fid_index = {x: t for t, x in enumerate(ids)}
        if fid not in self._fid_index:
            raise KeyError(f"no frame with id {fid!r}")
        return self._fid_index[fid]

    def indices_of(self, fids):
        """[index_of(f) for f in fids] (the cache is shared)."""
        return [self.index_of(f) for f in fids]

    # ---- writers (T semantics; codecs/ owns the encoding policy) ----
    # img: ONE entry per input kind (arrays / image bytes / file paths)
    def set_rgb(self, rgb):
        """a uint8 RGB ARRAY: (T,H,W,3) / an (H,W,3) frame (T=1). Each frame
        stored as jpg q<=95 (the policy in codecs.rgb). The frame COUNT must
        equal T. Image BYTES: set_rgb_bytes; FILE PATHS: set_rgb_paths."""
        self._store_img(normalize_img_array(rgb))

    def set_rgb_bytes(self, frames):
        """a sequence of image BYTES: flat (T,) (or (T,N) with N=1 for the
        /img layout). Per-frame policy in codecs.rgb (q<=95 verbatim, else
        re-encoded q95). Arrays: set_rgb; file paths: set_rgb_paths."""
        h, w, flat = normalize_img_bytes(frames)
        if len(flat) % self.T:                          # (T,N): N must be 1
            raise ValueError(f"rgb bytes must give T={self.T} frames "
                             f"(T*N), got {len(flat)}")
        self._store_img((h, w, flat[:self.T]))

    def set_rgb_paths(self, paths):
        """FILE PATHS (a str/Path or a flat (T,) sequence) read from disk,
        same per-frame policy as set_rgb_bytes. Arrays: set_rgb; bytes:
        set_rgb_bytes."""
        h, w, flat = normalize_img_paths(paths)
        if len(flat) % self.T:
            raise ValueError(f"rgb paths must give T={self.T} frames "
                             f"(T*N), got {len(flat)}")
        self._store_img((h, w, flat[:self.T]))

    def _store_img(self, hww):
        """(H, W, frames) -> /img/{0..T-1} (the frame count + size are
        checked; replace is non-atomic: a mid-loop failure leaves a partial
        img group — the temp-name+rename swap is a larger change than this
        call needs)."""
        h, w, frames = hww
        if len(frames) != self.T:
            raise ValueError(f"rgb gives {len(frames)} frames but T={self.T}")
        if (h, w) != (self.H, self.W):
            raise ValueError(f"rgb size ({h},{w}) != scene ({self.H},{self.W})")
        rg = self._h.require_group("img")
        for k in list(rg):
            del rg[k]
        for t, b in enumerate(frames):
            rg.create_dataset(f"{t}", data=np.frombuffer(b, np.uint8))

    def set_bgr(self, bgr):
        """(T,H,W,3) uint8 BGR array / an (H,W,3) array (T=1); ARRAYS ONLY
        (image bytes / file paths are rejected — the colour order on disk is
        unknowable, use set_rgb / set_rgb_bytes / set_rgb_paths). Channels
        reversed to RGB, then stored through the set_rgb policy (jpg q<=95)."""
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

    def _set_depth(self, depth, to_mm):
        d = self._cam(depth)
        d = to_mm(d)
        h, w = d.shape[1], d.shape[2]
        self._check_depth_hw("depth", h, w)
        self.write_section("depth", d, chunks=(1, h, w), **filter_args("depth"))

    def set_depth_mm(self, depth_mm):
        """(T, H, W) INTEGER mm / an (H, W) frame (T=1); MM ONLY (a float
        means meters — use set_depth_meters). May be stored smaller (T, h, w)
        if the aspect ratio H/W matches. gzip4 + shuffle (codecs.sections)."""
        self._set_depth(depth_mm, to_mm)

    def set_depth_meters(self, depth_m):
        """(T, H, W) float METERS / an (H, W) frame (T=1); METERS ONLY (an
        integer means mm — use set_depth_mm). d >= 65.535 m -> 0 (the invalid
        sentinel). Same (h, w) rules as set_depth_mm."""
        self._set_depth(depth_m, to_meters_mm)

    def set_depth_conf_01(self, conf):
        """(T, H, W) float 0-1 / an (H, W) frame (T=1); 0-1 FLOATS ONLY
        (integers are rejected — they are never a 0-1 confidence). Same (h, w)
        as depth; f16 0-1, gzip4 (no shuffle: most values 1.0)."""
        c = to_conf(self._cam(conf))
        self._check_depth_hw("depth_conf", c.shape[1], c.shape[2])
        self.write_section("depth_conf", c, chunks=(1, c.shape[1], c.shape[2]),
                          **filter_args("depth_conf"))

    def set_mask(self, mask):
        """bool/uint8 (T, H, W) per pixel / (H, W) broadcast / (T,) per-frame
        slots (whole-frame padding); full (H, W), gzip4."""
        m = to_mask(self._cam_mask(mask), self.T, self.H, self.W)
        self.write_section("mask", m, chunks=(1, self.H, self.W), **filter_args("mask"))

    def set_K(self, K):
        """(3,3) broadcast or (T, 3, 3) float OpenCV PINHOLE intrinsics
        (K[2,2] == 1, zero skew / K[1,0] / K[2,1] / K[2,0], fx, fy > 0 —
        check_K in codecs.pose; a pixel (u, v) centre is at (u, v)). Stored
        raw (no gzip — measured bigger with gzip for short sequences,
        break-even ~106 frames)."""
        self.write_section("K", check_K(broadcast_matrix(K, self.T, (3, 3))))

    def set_pose_c2w(self, pose_c2w):
        """(4,4) broadcast or (T, 4, 4) float camera-to-world (c2w) poses,
        x right / y down / z forward (last row [0,0,0,1], orthonormal R with
        |det R - 1| <= 1e-3 — check_pose in codecs.pose). Stored raw."""
        self.write_section("T", check_pose(broadcast_matrix(pose_c2w, self.T, (4, 4))))

    # ---- reads (T semantics; codecs/ owns the decoding policy) ----
    def rgb(self, t=0):
        """Frame t decoded: (H, W, 3) uint8 RGB."""
        self._t(t)
        if not self.has("img"):
            raise KeyError("no img frames")
        return decode_rgb(self.img_bytes(t))

    def img_bytes(self, t=0):
        """Frame t's stored jpg BYTES, undecoded (E4; cv2.imdecode(...,
        IMREAD_COLOR) reversed to RGB equals rgb(t))."""
        self._t(t)
        if not self.has("img"):
            raise KeyError("no img frames")
        return self._h["img"][f"{t}"][:].tobytes()

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
            # depth_conf must go first: it must share depth's (h, w), which
            # _check_depth_hw enforces, so the (h, w) comparison is always true
            # and only the presence of both is what matters.
            if name == "depth" and "depth" in h and "depth_conf" in h:
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

    # ---- per-frame / streaming write (eval request E5) ------------------------
    def allocate(self, sections=("depth", "K", "T"), depth_hw=None):
        """Pre-create the named sections ('img' or a SECTIONS name) for
        per-frame writing: write_frame(t, ...) then finalize(). A section
        already stored (a set_* ran) must be del_section()'d first; the
        img group is only marked (its per-frame datasets are created on
        write_frame). depth_hw: (h, w) for depth / depth_conf (default:
        the scene (H, W))."""
        if self._allocated is not None:
            raise ValueError("already allocated (write_frame / finalize)")
        names = list(sections)
        for name in names:
            if name == "img":
                continue
            if name not in SECTIONS:
                raise ValueError(f"unknown section {name!r}; known: "
                                 f"('img', {', '.join(repr(n) for n in SECTIONS)})")
            if name in self._h:
                raise ValueError(f"{name} is already stored: del_section "
                                 f"before allocate")
        self._allocated = set(names)
        self._written = {n: set() for n in names}
        for name in names:
            if name == "img":
                continue
            s = SECTIONS[name]
            if s.frame2d:
                hw = depth_hw if depth_hw is not None else (self.H, self.W)
                if not depth_hw_ok(*hw, self.H, self.W):
                    raise ValueError(f"depth_hw {hw} incompatible with scene "
                                     f"({self.H},{self.W}): aspect ratio must match")
                h_, w_ = hw
            else:
                h_, w_ = FRAME_SHAPE[name]
            self._h.create_dataset(name, shape=(self.T, h_, w_), dtype=s.dtype,
                                   fillvalue=s.fillvalue,
                                   chunks=(1, h_, w_), **filter_args(name))

    def write_frame(self, t, rgb=None, depth_mm=None, depth_meters=None,
                    depth_conf=None, mask=None, K=None, pose_c2w=None):
        """Write ONE frame into the allocated sections (allocate first):
        the frame's rgb (an (H,W,3) uint8 array or its jpg bytes) and / or
        depth_mm | depth_meters (one of the two — the unit is in the name),
        depth_conf (0-1 float), mask, K ((3,3) pinhole-checked), pose_c2w
        ((4,4) c2w-checked) — same rules as the set_* methods, one frame at
        a time (peak memory is O(one frame)). A section not allocated raises."""
        if self._allocated is None:
            raise ValueError("allocate(sections=...) before write_frame")
        self._t(t)
        if any(t in s for s in self._written.values()):
            raise ValueError(f"frame {t} already written (it cannot be "
                             f"written twice)")
        wrote = False
        if rgb is not None:
            self._require_allocated("img")
            h, w, flat = (normalize_img_bytes([bytes(rgb)])
                          if isinstance(rgb, (bytes, bytearray, memoryview))
                          else normalize_img_array(rgb))
            if (h, w) != (self.H, self.W):
                raise ValueError(f"rgb size ({h},{w}) != scene ({self.H},{self.W})")
            self._h.require_group("img").create_dataset(
                f"{t}", data=np.frombuffer(flat[0], np.uint8))
            self._written["img"].add(t)
            wrote = True
        if depth_mm is not None and depth_meters is not None:
            raise ValueError("write_frame takes one of depth_mm / depth_meters "
                             "(the unit is in the name) — not both")
        if depth_mm is not None:
            self._require_allocated("depth")
            a = np.asarray(depth_mm, dtype=np.uint16)
            d = to_mm(a[None] if a.ndim == 2 else a)
            self._write_frame_data("depth", t, d)
            wrote = True
        elif depth_meters is not None:
            self._require_allocated("depth")
            a = np.asarray(depth_meters, dtype=np.float64)
            d = to_meters_mm(a[None] if a.ndim == 2 else a)
            self._write_frame_data("depth", t, d)
            wrote = True
        if depth_conf is not None:
            self._require_allocated("depth_conf")
            a = np.asarray(depth_conf, dtype=np.float64)
            c = to_conf(a[None] if a.ndim == 2 else a)
            self._write_frame_data("depth_conf", t, c)
            wrote = True
        if mask is not None:
            self._require_allocated("mask")
            a = np.asarray(mask)
            if a.ndim == 2:
                if a.shape != (self.H, self.W):
                    raise ValueError(f"mask frame must be (H, W) = "
                                     f"({self.H},{self.W}), got {a.shape}")
                m = to_mask(a[None], 1, self.H, self.W)
            elif a.ndim == 3 and a.shape[0] == 1:
                m = to_mask(a, 1, self.H, self.W)
            else:
                raise ValueError(f"mask frame must be (H, W) or (1, H, W), "
                                 f"got {a.shape}")
            self._write_frame_data("mask", t, m)
            wrote = True
        if K is not None:
            self._require_allocated("K")
            self._h["K"][t] = check_K(np.asarray(K, dtype=np.float32))
            self._written["K"].add(t)
            wrote = True
        if pose_c2w is not None:
            self._require_allocated("T")
            self._h["T"][t] = check_pose(np.asarray(pose_c2w, dtype=np.float32))
            self._written["T"].add(t)
            wrote = True
        if not wrote:
            raise ValueError("write_frame got nothing to write (pass at least "
                             "one of rgb / depth_mm / depth_meters / depth_conf / "
                             "mask / K / pose_c2w)")

    def _require_allocated(self, name):
        if name not in self._allocated:
            raise ValueError(f"section {name!r} is not allocated "
                             f"(allocate(sections=...))")

    def _write_frame_data(self, name, t, d1):
        """one (1, h, w) converted frame row into an allocated dataset"""
        d = self._h[name]
        h_, w_ = d.shape[1:]
        if d1.shape[1:] != (h_, w_):
            raise ValueError(f"{name} frame shape {d1.shape[1:]} != allocated "
                             f"({h_},{w_})")
        d[t] = d1[0]
        self._written[name].add(t)

    def finalize(self):
        """Check that every t was written into every allocated section (a
        skipped t raises); clears the streaming state (the file is then
        plain set_*/read-able)."""
        if self._allocated is None:
            raise ValueError("nothing was allocated")
        missing = {n: sorted(set(range(self.T)) - s)
                   for n, s in self._written.items() if s != set(range(self.T))}
        if missing:
            raise ValueError(f"frames not written: "
                             + "; ".join(f"{n}: {s[:5]}{' …' if len(s) > 5 else ''}"
                                         for n, s in sorted(missing.items())))
        self._allocated, self._written = None, set()
        return self
