"""H5CamRig: N H5Cam files -> one multi-camera reference file (T, N, Hmax,
Wmax) — REFERENCES only, nothing copied into the rig file (verified
2026-11-13).

- STRICT checks: same T, same is_metric, same section set (depth/mask/K/T/
  depth_conf), rgb per-frame size == declared (H, W), depth/depth_conf/mask
  per-frame size == frame 0, and H or W identical across ALL cameras (the
  other axis = max; both axes differ -> rejected, no file written).
- /rgb: ExternalLinks to each camera's own /rgb/{t} — 0 bytes in the rig
  file, no copy, no re-encode; rgb(t, n) therefore decodes at camera n's
  NATIVE (H, W), not the common (Hmax, Wmax).
- /depth /mask /K /T /depth_conf: VIRTUAL datasets referencing the camera
  files — 0 bytes on disk in the rig file, the cameras' compressed data
  mapped directly, at (T, N, Hmax, Wmax). Smaller cameras: per-frame blocks
  (h, w) -> (t, n, :h, :w) with fillvalue covering the pad region
  (0 = invalid / False = padding) — matches the stored sentinels, so they
  read back at the common (Hmax, Wmax); depth/depth_conf/mask use the _io
  decode helpers (resize to (Hmax, Wmax), sentinel re-clip).
- attrs.cam_files (absolute) + attrs.cam_hw (per-camera (H, W));
  H5CamRig(open) verifies the camera files exist, because a missing/renamed
  source makes VDS reads SILENTLY return the fillvalue (and ExternalLinks
  raise KeyError) instead of erroring clearly.
- the result is also a plain H5Base file (attrs.TNHW, /rgb t_n keys, /meta),
  openable directly with H5Base for the raw handle.
"""
import os

import h5py
import numpy as np

from ._io import (depth_conf_to_f32, depth_mask, depth_to_meters, depth_to_mm,
                  decode_rgb, has, tags_dict)
from .h5base import H5Base
from .h5cam import H5Cam

_STR = h5py.string_dtype()

# name, vds dtype, fillvalue, per-frame shape: (H, W) for 2-D sections,
# the FIXED (3,3)/(4,4) for K/T (every camera shares that frame shape)
_SECTIONS = (("depth", np.uint16, 0, (2,)),
             ("mask", np.bool_, False, (2,)),
             ("K", np.float32, 0.0, (3, 3)),
             ("T", np.float32, 0.0, (4, 4)),
             ("depth_conf", np.float16, 0.0, (2,)))


class H5CamRig(H5Base):
    """N H5Cam files -> one multi-camera file (T, N, Hmax, Wmax) of
    references only — the cameras' SEQS merged into one time axis (the rig
    has no seq: it IS the merge). Build: H5CamRig(out_path, cam_paths, suffix=...)
    — all cameras must share the SAME dataset and scene (checked); suffix is
    the rig's own variant marker (default empty). open:
    H5CamRig(out_path) — verifies attrs.cam_files exist. All scene reads
    (rgb/depth/mask/K/pose/depth_conf/depth_mask + meta) are (T, N) methods
    on this class, built on the _io decode helpers."""

    # (name -> expected type); the rig has NO "seq" — it IS the merged axis
    ATTR_KEYS = {"TNHW": np.ndarray,       # int64 (4,) = (T, N, Hmax, Wmax)
                 "is_metric": bool,
                 "cam_files": np.ndarray,  # str[] absolute source cam files
                 "cam_hw": np.ndarray}     # int64 (N, 2) per-camera (H, W)
    META_KEYS = {"dataset": str,           # top-level dataset name (must match across cams)
                 "scene": str,             # scene id (must match across cams)
                 "suffix": str,            # this rig's own variant marker
                 "split": str,
                 "meta_json": str,
                 "frame_indices": np.ndarray,  # int64 (T,)
                 "tags": dict}

    def __init__(self, out_path, cam_paths=None, suffix=""):
        if cam_paths is None:            # open
            super().__init__(out_path)
            a = self._h
            if "TNHW" not in a.attrs or "cam_files" not in a.attrs:
                self.close()
                raise ValueError(f"{self.path}: not an H5CamRig file (no attrs.TNHW / cam_files)")
            missing = [p for p in a.attrs["cam_files"] if not os.path.exists(p)]
            if missing:
                self.close()
                raise FileNotFoundError(
                    f"{self.path}: missing camera source file(s) — VDS would "
                    f"silently read fillvalues and ExternalLinks raise: {missing}")
            self.T, self.N, self.H, self.W = (int(x) for x in a.attrs["TNHW"])
            return
        self._build(out_path, cam_paths, suffix)

    @property
    def shape(self):
        return self.T, self.N, self.H, self.W

    @property
    def cam_files(self):
        """List[str] of the source camera files (absolute) this rig references."""
        return [str(p) for p in self._h.attrs["cam_files"]]

    @property
    def cam_hw(self):
        """List[(H, W)] of each camera's native resolution, in cam order."""
        return [tuple(int(x) for x in row) for row in self._h.attrs["cam_hw"]]

    @property
    def is_metric(self):
        v = self.attr("is_metric")
        if v is None:
            m = self.meta("is_metric")
            return bool(m) if m is not None else True
        return bool(v)

    def has(self, name):
        return has(self._h, name)

    # ---- scene reads (T, N semantics; _io owns the decode policy) ----
    def _t(self, t, n):
        if not (0 <= t < self.T and 0 <= n < self.N):
            raise IndexError(f"frame ({t},{n}) out of range (T={self.T}, N={self.N})")

    def _f(self, name, t, n):
        self._t(t, n)
        if not self.has(name):
            raise KeyError(f"no {name} section")
        return self._h[name][t, n][:]

    def rgb(self, t=0, n=0):
        """Frame (t, n) decoded: camera n's NATIVE (H, W, 3) uint8 RGB (the
        /rgb entry is an ExternalLink to that camera's /rgb/{t})."""
        self._t(t, n)
        if not self.has("rgb"):
            raise KeyError("no rgb frames")
        return decode_rgb(self._h["rgb"][f"{t}_{n}"][:])

    def depth(self, t=0, n=0, meters=False):
        """Frame (t, n) depth: (Hmax, Wmax) uint16 mm (default) or float32
        meters; pad rows/cols (the VDS fillvalue region) read back as 0."""
        f = self._f("depth", t, n)
        return depth_to_meters(f, self.H, self.W) if meters else depth_to_mm(f, self.H, self.W)

    def depth_conf(self, t=0, n=0):
        """Frame (t, n) confidence: (Hmax, Wmax) f32 0-1; pad reads back 0.0."""
        return depth_conf_to_f32(self._f("depth_conf", t, n), self.H, self.W)

    def mask(self, t=0, n=0):
        """Frame (t, n) mask: (Hmax, Wmax) bool; the pad region reads False
        (the VDS fillvalue)."""
        return self._f("mask", t, n)

    def K(self, t=0, n=0):
        return self._f("K", t, n)

    def pose(self, t=0, n=0):
        return self._f("T", t, n)

    def depth_mask(self, t=0, n=0, conf_min=None):
        """Frame (t, n) usable-depth mask: (Hmax, Wmax) bool — depth not an
        invalid sentinel (0 AND 65535 excluded; pad rows/cols are 0) and, if
        depth_conf present, conf > (0 if conf_min is None, else conf_min)."""
        dm = depth_to_mm(self._f("depth", t, n), self.H, self.W)
        conf = (depth_conf_to_f32(self._f("depth_conf", t, n), self.H, self.W)
                if self.has("depth_conf") else None)
        return depth_mask(dm, conf, conf_min)

    @property
    def dataset(self):
        return self.meta("dataset")

    @property
    def scene(self):
        return self.meta("scene")

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

    # ---- build ----
    def _build(self, out_path, cam_paths, suffix):
        self.path = str(out_path)
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
            ds = {c.dataset for c in cams}
            ss = {c.scene for c in cams}
            if len(ds) > 1:
                raise ValueError(f"camera dataset mismatch (must merge the same scene): {sorted(ds, key=repr)}")
            if len(ss) > 1:
                raise ValueError(f"camera scene mismatch (must merge the same scene): {sorted(ss, key=repr)}")
            dataset, scene = (ds.pop() if ds else None), (ss.pop() if ss else None)
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
                    img = decode_rgb(c._h["rgb"][f"{t}"][:])
                    if img.shape[:2] != (c.H, c.W):
                        raise ValueError(f"cam {n} rgb frame {t} size {img.shape[:2]} "
                                         f"!= declared ({c.H},{c.W})")
                for name, *_ in _SECTIONS:
                    if name in secs and name not in ("K", "T"):
                        s0 = c._h[name][0].shape
                        for t in range(1, T):
                            if c._h[name][t].shape != s0:
                                raise ValueError(f"cam {n} {name} frame {t} shape "
                                                 f"{c._h[name][t].shape} != frame 0 {s0}")

            with h5py.File(self.path, "w", libver="latest") as h:
                h.attrs["TNHW"] = np.array((T, N, H, W), np.int64)
                h.attrs["is_metric"] = is_metric
                abs_paths = [os.path.abspath(c.path) for c in cams]
                h.attrs["cam_files"] = np.array(abs_paths, dtype=_STR)
                h.attrs["cam_hw"] = np.array([(c.H, c.W) for c in cams], np.int64)
                # /rgb: nothing stored — each (t, n) is an ExternalLink to the
                # camera's own /rgb/{t} (variable-length jpg cannot be a
                # virtual dataset). rgb(t, n) decodes at camera n's NATIVE (H, W).
                rg = h.create_group("rgb")
                for n, c in enumerate(cams):
                    for t in range(T):
                        rg[f"{t}_{n}"] = h5py.ExternalLink(abs_paths[n], f"/rgb/{t}")
                # fixed-shape sections: virtual datasets referencing the cameras
                for name, dtype, fill, fshape in _SECTIONS:
                    if name not in secs:
                        continue
                    fixed = name in ("K", "T")            # (3,3)/(4,4): every cam identical
                    shp = (H, W) if not fixed else fshape
                    lay = h5py.VirtualLayout(shape=(T, N, *shp), dtype=dtype)
                    for n, c in enumerate(cams):
                        if fixed:                        # (3,3)/(4,4) frames: one block per camera
                            lay[:, n, ...] = h5py.VirtualSource(
                                abs_paths[n], name, shape=(T, *c._h[name][0].shape))
                        elif (c.H, c.W) == (H, W) and c._h[name][0].shape == (H, W):
                            lay[:, n, ...] = h5py.VirtualSource(        # full-size: one block
                                abs_paths[n], name, shape=(T, H, W))
                        else:  # per-frame (h, w) blocks; pad region -> fillvalue
                            for t in range(T):
                                h_, w_ = c._h[name][t].shape
                                lay[t, n, :h_, :w_] = h5py.VirtualSource(
                                    abs_paths[n], name, shape=(h_, w_))
                    h.create_virtual_dataset(name, lay, fillvalue=fill)
                m = h.require_group("meta")
                for k, v in (("dataset", dataset), ("scene", scene), ("suffix", suffix)):
                    if v:
                        m.create_dataset(k, data=np.array(str(v), dtype=_STR))
        finally:
            for c in cams:
                c.close()
        self._h = h5py.File(self.path, "r")
        self.T, self.N, self.H, self.W = (int(x) for x in self._h.attrs["TNHW"])
