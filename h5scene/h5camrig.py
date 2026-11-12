"""H5CamRig: N H5Cam files -> one multi-camera reference file (T, N, Hmax,
Wmax) — REFERENCES only, nothing copied into the rig file (verified
2026-11-13).

- STRICT build checks (all run BEFORE any file is written): same T, same
  is_metric, SAME dataset and scene across all cameras (the rig merges the
  cameras' seqs into ONE time axis — it has no seq of its own), same section
  set (rgb in ALL or in NONE), rgb per-frame size == the declared (H, W)
  (checked from the jpg SOF header, no full decode), depth/depth_conf stored
  at the camera's NATIVE (H, W) (downsampled depth is rejected — it cannot be
  upscaled through a VDS block), and H or W identical across ALL cameras
  (the other axis = max, top-left aligned; both axes differ -> rejected, no
  file written).
- build is write-temp-then-rename: a failed build never touches an existing
  output file (and the temp file is removed).
- /rgb: ExternalLinks to each camera's own /rgb/{t} — 0 bytes in the rig
  file, no copy, no re-encode; rgb(t, n) therefore decodes at camera n's
  NATIVE (H, W), not the common (Hmax, Wmax).
- /depth /mask /K /T /depth_conf: VIRTUAL datasets referencing the camera
  files — 0 bytes on disk in the rig file, the cameras' compressed data
  mapped directly, at (T, N, Hmax, Wmax). A smaller camera: per-frame blocks
  (h, w) -> (t, n, :h, :w) with fillvalue covering the pad region
  (0 = invalid / False = padding) — matches the stored sentinels, so they
  read back at the common (Hmax, Wmax).
- attrs.cam_files (paths RELATIVE to the rig file's directory — VDS and
  ExternalLinks resolve that way, so a rig copied/moved together with its
  cameras keeps working) + attrs.cam_hw (per-camera (H, W)) + attrs.cam_fp
  (per-camera (T, H, W, size) CONTENT fingerprint — THW is read from the
  camera at open; legacy (T, H, W, size, mtime_ns) rows still verified):
  H5CamRig(open) verifies the camera files still exist AND match the
  fingerprint, because a missing/renamed/rewritten source makes VDS reads
  SILENTLY return the fillvalue (and ExternalLinks raise KeyError) instead
  of erroring clearly.
- the result is also a plain H5Base file (attrs.TNHW, /rgb t_n keys, /meta).
"""
import os
import os.path as osp

import h5py
import numpy as np

from .codecs import (FRAME_SHAPE, SECTIONS, conf_frame, depth_mask as _dm,
                     decode_rgb, has, is_metric as _is_metric,
                     jpeg_hw, mm_frame, meters_frame)
from .h5base import H5Base
from .h5cam import H5Cam

_STR = h5py.string_dtype(encoding="utf-8")


class H5CamRig(H5Base):
    """N H5Cam files -> one multi-camera file (T, N, Hmax, Wmax) of
    references only — the cameras' seqs merged into one time axis (the rig
    has no seq: it IS the merge). Build: H5CamRig.build(out, cam_paths,
    suffix=...) or H5CamRig(out, cam_paths, suffix=...) — all cameras must
    share the SAME dataset and scene (checked); suffix is the rig's own
    variant marker (default empty). Open: H5CamRig.open(path) — verifies the
    camera files exist and match the stored fingerprint. All scene reads
    (rgb/depth/mask/K/pose/depth_conf/depth_mask + meta) are (T, N) methods
    on this class, built on the codecs/ decode helpers."""

    # (name -> expected type); the rig has NO "seq" and NO "src_frame_idxs" —
    # it IS the merged time axis (the cameras' per-source frame numbers have
    # no single meaning for the merged axis). META_KEYS is DERIVED from
    # H5Cam's (the source camera's registry): cam keys minus "seq" and
    # "src_frame_idxs", plus the rig's own "suffix".
    ATTR_KEYS = {"TNHW": np.ndarray,       # int64 (4,) = (T, N, Hmax, Wmax)
                 "is_metric": bool,
                 "cam_files": np.ndarray,  # str[] source cam files, relative to the rig
                 "cam_hw": np.ndarray,     # int64 (N, 2) per-camera (H, W)
                 "cam_fp": np.ndarray}     # int64 (N, 4) = (T, H, W, size) content fp
    META_KEYS = {**{k: v for k, v in H5Cam.META_KEYS.items()
                    if k not in ("seq", "src_frame_idxs")},
                 "suffix": str}            # this rig's own variant marker

    @classmethod
    def build(cls, out_path, cam_paths, suffix=""):
        return cls(out_path, cam_paths, suffix=suffix)

    @classmethod
    def open(cls, path, mode="r"):
        return cls(path, mode=mode)

    def __init__(self, out_path, cam_paths=None, suffix="", mode=None):
        if cam_paths is not None:            # build
            self.path = str(out_path)
            self._build(self.path, cam_paths, suffix)
            self.T, self.N, self.H, self.W = (int(x) for x in self._h.attrs["TNHW"])
            return
        super().__init__(out_path, mode=mode)
        try:
            a = self._h
            if "TNHW" not in a.attrs or "cam_files" not in a.attrs:
                raise ValueError(f"{self.path}: not an H5CamRig file "
                                 "(no attrs.TNHW / cam_files)")
            self._verify_sources()
            self.T, self.N, self.H, self.W = (int(x) for x in a.attrs["TNHW"])
        except Exception:
            self.close()
            raise

    # ---- source fingerprint (a rewritten/missing camera is caught at open) ----
    def _verify_sources(self):
        a = self._h.attrs
        # cam_files is stored relative to the rig file's directory (VDS and
        # ExternalLinks resolve the same way) — a rig moved together with its
        # cameras keeps working; legacy absolute paths stay usable too.
        d = osp.dirname(osp.abspath(self.path))
        files = [osp.normpath(osp.join(d, str(p))) for p in a["cam_files"]]
        missing = [p for p in files if not osp.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{self.path}: missing camera source file(s) — VDS would "
                f"silently read fillvalues and ExternalLinks raise: {missing}")
        T = int(a["TNHW"][0])                       # the rig's own T
        hw = [(int(a["cam_hw"][i][0]), int(a["cam_hw"][i][1])) for i in range(len(files))]
        for i, p in enumerate(files):
            st = os.stat(p)
            if "cam_fp" in a:
                fp = a["cam_fp"][i]
                if fp.shape[0] == 5:                # legacy (T, H, W, size, mtime_ns)
                    Tc, Hc, Wc, size, mtime = (int(x) for x in fp)
                    bad = ((st.st_size, st.st_mtime_ns) != (size, mtime)
                           or (Tc, Hc, Wc) != (T, hw[i][0], hw[i][1]))
                    why = "size/mtime"
                else:                               # (T, H, W, size): the camera's own
                    Tc, Hc, Wc, size = (int(x) for x in fp)   # THW is read and compared
                    with h5py.File(p, "r") as c:
                        thw = c.attrs.get("THW")
                    cam = None if thw is None else tuple(int(x) for x in thw)
                    bad = (size != st.st_size or cam != (Tc, Hc, Wc)
                           or (Tc, Hc, Wc) != (T, hw[i][0], hw[i][1]))
                    why = "THW or size"
                if bad:
                    raise ValueError(
                        f"{self.path}: camera {i} ({p}) no longer matches the "
                        f"fingerprint stored at build time ({why} changed) "
                        f"— the VDS would misread")

    # ---- properties ----
    @property
    def shape(self):
        return self.T, self.N, self.H, self.W

    def __len__(self):
        return self.T

    @property
    def sections(self):
        return {n for n in ("rgb", *SECTIONS) if self.has(n)}

    def __repr__(self):
        return (f"<{type(self).__name__} {self.path} shape={self.shape} "
                f"sections={sorted(self.sections)}>")

    @property
    def cam_files(self):
        """List[str] of the source camera files (absolute) this rig references
        (stored relative to the rig file's directory)."""
        d = osp.dirname(osp.abspath(self.path))
        return [osp.normpath(osp.join(d, str(p))) for p in self._h.attrs["cam_files"]]

    @property
    def cam_hw(self):
        """List[(H, W)] of each camera's native resolution, in cam order."""
        return [tuple(int(x) for x in row) for row in self._h.attrs["cam_hw"]]

    @property
    def is_metric(self):
        return _is_metric(self._h)

    def has(self, name):
        return has(self._h, name)

    # ---- scene reads (T, N semantics; codecs/ owns the decode policy) ----
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
        return decode_rgb(self._h["rgb"][f"{t}_{n}"][:].tobytes())

    def depth(self, t=0, n=0, meters=False):
        """Frame (t, n) depth: (Hmax, Wmax) uint16 mm (default) or float32
        meters; pad rows/cols (the VDS fillvalue region) read back as 0."""
        f = self._f("depth", t, n)
        return meters_frame(f, self.H, self.W) if meters else mm_frame(f, self.H, self.W)

    def depth_conf(self, t=0, n=0):
        """Frame (t, n) confidence: (Hmax, Wmax) f32 0-1; pad reads back 0.0."""
        return conf_frame(self._f("depth_conf", t, n), self.H, self.W)

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
        mm = mm_frame(self._f("depth", t, n), self.H, self.W)
        conf = (conf_frame(self._f("depth_conf", t, n), self.H, self.W)
                if self.has("depth_conf") else None)
        return _dm(mm, conf, conf_min)

    # ---- meta (this class's own keys, on top of H5Base.meta) ----
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
    def tags(self):
        return self.meta("tags")

    # ---- build (validate first, write temp, rename; never touch out on error) ----
    def _build(self, out_path, cam_paths, suffix):
        # 0) the paths themselves
        if not cam_paths:
            raise ValueError("need at least one camera path")
        out_path = osp.abspath(str(out_path))
        absp = [osp.abspath(str(p)) for p in cam_paths]
        if len(set(absp)) != len(absp):
            raise ValueError(f"duplicate camera paths: {sorted(set(absp))}")
        if out_path in absp:
            raise ValueError(f"the output path {out_path!r} is one of its own "
                             f"cameras — the build would overwrite the camera")
        missing = [p for p in absp if not osp.exists(p)]
        if missing:
            raise FileNotFoundError(f"camera source file(s) missing: {missing}")
        os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
        # 1) open + validate everything (no file written yet)
        cams = [H5Cam(p) for p in absp]
        try:
            vals, meta = self._validate(cams)
        except Exception:
            for c in cams:
                c.close()
            raise
        # 2) write to a temp file, then rename into place (the cameras are
        # always released, even when this fails)
        tmp = f"{out_path}.tmp.{os.getpid()}"
        try:
            try:
                self._write(tmp, absp, vals, meta, suffix)
                os.replace(tmp, out_path)
            finally:
                try:
                    if osp.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
        finally:
            for c in cams:
                c.close()
        self._h = h5py.File(out_path, "r")

    def _validate(self, cams):
        """All cross-camera checks; returns (T, H, W, N, has_rgb, secs, dhw,
        (dataset, scene, shared_meta))."""
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
            raise ValueError(f"camera dataset mismatch (must merge the same scene): "
                             f"{sorted(ds, key=repr)}")
        if len(ss) > 1:
            raise ValueError(f"camera scene mismatch (must merge the same scene): "
                             f"{sorted(ss, key=repr)}")
        dataset, scene = ds.pop(), ss.pop()
        hs = {c.H for c in cams}
        ws = {c.W for c in cams}
        if len(hs) > 1 and len(ws) > 1:
            raise ValueError(f"rig needs H or W identical across cameras; "
                             f"got H={sorted(hs)} W={sorted(ws)}")
        H, W, N = max(hs), max(ws), len(cams)
        has_rgb = [c.has("rgb") for c in cams]
        if any(has_rgb) and not all(has_rgb):
            raise ValueError(f"rgb must be present in ALL cameras or in NONE "
                             f"(got {has_rgb})")
        secs = [tuple(n for n in SECTIONS if n in c._h) for c in cams]
        if len(set(secs)) != 1:
            raise ValueError(f"camera section sets differ: {secs}")
        secs = secs[0]
        dhw = []
        for n, c in enumerate(cams):
            h_ = None
            for name in ("depth", "depth_conf"):
                if name in c._h:
                    s = c._h[name].shape[1:]
                    if s != (c.H, c.W):
                        raise ValueError(f"cam {n} {name} is stored at {tuple(s)} "
                                         f"but the camera is ({c.H}, {c.W}): the rig "
                                         f"maps per-frame VDS blocks and cannot "
                                         f"upscale downsampled depth")
                    h_ = s
            if has_rgb[n]:                        # size check from the SOF header
                for t in range(c.T):
                    b = c._h["rgb"][f"{t}"][:].tobytes()
                    hw = jpeg_hw(b)
                    if hw is None:
                        hw = decode_rgb(b).shape[:2]
                    if hw != (c.H, c.W):
                        raise ValueError(f"cam {n} rgb frame {t} size {hw} "
                                         f"!= declared ({c.H},{c.W})")
            dhw.append((c.H, c.W))
        # shared meta: inherited only when EVERY camera agrees (no None mixed in)
        shared = {}
        for k in ("split", "tags"):
            vals = [c.meta(k) for c in cams]
            if all(v is not None for v in vals) and len({repr(v) for v in vals}) == 1:
                shared[k] = vals[0]
        return (T, H, W, N, has_rgb[0], secs, dhw), (dataset, scene, is_metric, shared)

    def _write(self, path, absp, vals, meta, suffix):
        T, H, W, N, has_rgb, secs, dhw = vals
        dataset, scene, is_metric, shared = meta
        # references are stored relative to the rig file's directory: VDS and
        # ExternalLinks resolve that way, so a rig copied/moved together with
        # its cameras keeps working (the fingerprint is content-based, not
        # mtime-based, so `cp`/`rsync`/git-checkout don't break it).
        d = osp.dirname(path) or "."
        relp = [osp.relpath(p, d) for p in absp]
        with h5py.File(path, "w", libver="latest") as h:
            h.attrs["TNHW"] = np.array((T, N, H, W), np.int64)
            h.attrs["is_metric"] = is_metric
            h.attrs["cam_files"] = np.array(relp, dtype=_STR)
            h.attrs["cam_hw"] = np.array(dhw, np.int64)
            fps = []
            for p, (hh, ww) in zip(absp, dhw):
                st = os.stat(p)
                fps.append((T, hh, ww, st.st_size))    # (T, H, W, size): content
            h.attrs["cam_fp"] = np.array(fps, np.int64)
            # /rgb: nothing stored — each (t, n) is an ExternalLink to the
            # camera's own /rgb/{t} (variable-length jpg cannot be a VDS).
            # rgb(t, n) decodes at camera n's NATIVE (H, W).
            if has_rgb:
                rg = h.create_group("rgb")
                for n, r in enumerate(relp):
                    for t in range(T):
                        rg[f"{t}_{n}"] = h5py.ExternalLink(r, f"/rgb/{t}")
            # fixed-shape sections: VDS referencing the cameras
            for name in secs:
                s = SECTIONS[name]
                fixed = not s.frame2d               # (3,3)/(4,4) frames
                shp = FRAME_SHAPE[name] if fixed else (H, W)
                lay = h5py.VirtualLayout(shape=(T, N, *shp), dtype=s.dtype)
                for n, (hh, ww) in enumerate(dhw):
                    if fixed:                       # (3,3)/(4,4): one block per camera
                        lay[:, n, ...] = h5py.VirtualSource(
                            relp[n], name, shape=(T, *FRAME_SHAPE[name]))
                    elif (hh, ww) == (H, W):       # full-size: one block
                        lay[:, n, ...] = h5py.VirtualSource(relp[n], name, shape=(T, H, W))
                    else:                           # per-frame (h, w) blocks;
                        # pad region -> fillvalue; the [t] selects camera frame t
                        for t in range(T):
                            lay[t, n, :hh, :ww] = h5py.VirtualSource(
                                relp[n], name, shape=(T, hh, ww))[t]
                h.create_virtual_dataset(name, lay, fillvalue=s.fillvalue)
            m = h.require_group("meta")
            for k, v in (("dataset", dataset), ("scene", scene), ("suffix", suffix)):
                if v:
                    m.create_dataset(k, data=np.array(str(v), dtype=_STR))
            for k, v in shared.items():
                if isinstance(v, dict):
                    tg = m.create_group(k)
                    for kk, vv in v.items():
                        tg.create_dataset(kk, data=np.array(vv, dtype=_STR))
                elif isinstance(v, np.ndarray):
                    m.create_dataset(k, data=v.astype(np.int64))
                else:
                    m.create_dataset(k, data=np.array(str(v), dtype=_STR))
