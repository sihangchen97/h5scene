"""H5CamRig: N H5Cam files -> one multi-camera reference file (T, N, Hmax,
Wmax) — REFERENCES only, nothing copied into the rig file (verified
2026-11-13).

- STRICT build checks (all run BEFORE any file is written): same T, same
  is_metric, SAME dataset and scene across all cameras (the rig merges the
  cameras' seqs into ONE time axis — it has no seq of its own), same section
  set (img in ALL or in NONE), img per-frame size == the declared (H, W)
  (checked from the jpg SOF header, no full decode), depth/depth_conf stored
  at the camera's NATIVE (H, W) (downsampled depth is rejected — it cannot be
  upscaled through a VDS), and H or W identical across ALL cameras (the
  other axis = max, top-left aligned; both axes differ -> rejected, no
  file written).
- build is write-temp-then-rename: a failed build never touches an existing
  output file (and the temp file is removed).
- /img: ExternalLinks to each camera's own /img/{t} — 0 bytes in the rig
  file, no copy, no re-encode; rgb(t, n) therefore decodes at camera n's
  NATIVE (H, W), not the common (Hmax, Wmax).
- /depth /mask /K /T /depth_conf: VIRTUAL datasets referencing the camera
  files — 0 bytes on disk in the rig file, the cameras' compressed data
  mapped directly (one mapping per camera), at (T, N, Hmax, Wmax). A
  smaller camera: its (h, w) part is mapped to (t, n, :h, :w) with the
  fillvalue covering the pad region (0 = invalid / False = padding) —
  matches the stored sentinels, so they read back at the common
  (Hmax, Wmax).
- attrs.cam_files (paths RELATIVE to the rig file's directory — VDS and
  ExternalLinks resolve that way, so a rig copied/moved together with its
  cameras keeps working) + attrs.cam_hw (per-camera (H, W)) + attrs.cam_fp
  (per-camera (T, H, W) — the camera's own THW, read back from the camera
  at open): H5CamRig(open) verifies the camera files still exist, match
  that THW, AND still hold every mapped section with the mapped shape
  (a del_section / re-store on a camera leaves its own THW unchanged, so
  only the section check catches it), because a missing/renamed/rewritten
  source makes VDS reads SILENTLY return the fillvalue (and ExternalLinks
  raise KeyError) instead of erroring clearly. The fingerprint is the THW,
  NOT the file size — a metadata edit on a camera no longer breaks open.
- the result is also a plain H5Base file (attrs.TNHW, /img t_n keys, /meta).
"""
import os
import os.path as osp

import h5py
import numpy as np

from .codecs import (FRAME_SHAPE, SECTIONS, conf_frame, depth_mask as _dm,
                     decode_rgb, has, is_metric as _is_metric,
                     jpeg_hw, mm_frame, meters_frame)
from .h5base import H5Base, _STR
from .h5cam import H5Cam


class H5CamRig(H5Base):
    """N H5Cam files -> one multi-camera file (T, N, Hmax, Wmax) of
    references only — the cameras' seqs merged into one time axis (the rig
    has no seq: it IS the merge). Build: H5CamRig.build(out, cam_paths,
    suffix=...) — all cameras must share the SAME dataset and scene (checked);
    suffix is the rig's own variant marker (default empty). Open:
    H5CamRig.open(path) verifies the camera files exist and match the stored
    fingerprint. build/open are the only constructors (__init__ opens only).
    All scene reads (rgb/depth/mask/K/pose/depth_conf/depth_mask + meta) are
    (T, N) methods
    on this class, built on the codecs/ decode helpers."""

    # (name -> expected type); the rig has NO "seq" and NO "src_frame_idxs" —
    # it IS the merged time axis (the cameras' per-source frame numbers have
    # no single meaning for the merged axis). META_KEYS is DERIVED from
    # H5Cam's (the source camera's registry): cam keys minus "seq" and
    # "src_frame_idxs"; the "suffix" is H5Cam's own.
    ATTR_KEYS = {"TNHW": np.ndarray,       # int64 (4,) = (T, N, Hmax, Wmax)
                 "is_metric": bool,
                 "cam_files": np.ndarray,  # str[] source cam files, relative to the rig
                 "cam_hw": np.ndarray,     # int64 (N, 2) per-camera (H, W)
                 "cam_fp": np.ndarray}     # int64 (N, 3) = (T, H, W) content fp
    META_KEYS = {k: v for k, v in H5Cam.META_KEYS.items()
                 if k not in ("seq", "src_frame_idxs")}   # the suffix is already there

    @classmethod
    def build(cls, out_path, cam_paths, suffix=""):
        """Build a rig from the camera files (the ONLY construction path;
        a failure leaves no half-made file and an existing output untouched)."""
        rig = object.__new__(cls)            # bypass the open-only __init__
        rig.path = str(out_path)
        rig._created = False
        rig._build(rig.path, cam_paths, suffix)
        rig.T, rig.N, rig.H, rig.W = (int(x) for x in rig._h.attrs["TNHW"])
        return rig

    @classmethod
    def open(cls, path, mode="r"):
        return cls(path, mode=mode)

    def __init__(self, out_path, mode="r"):
        """Open an existing rig file ("r" or "r+"). Building goes through
        H5CamRig.build — not this."""
        super().__init__(out_path, mode=mode)
        try:
            a = self._h
            if any(k not in a.attrs for k in
                   ("TNHW", "cam_files", "cam_hw", "cam_fp", "is_metric")):
                raise ValueError(f"{self.path}: not an H5CamRig file "
                                 "(no attrs.TNHW / cam_files / cam_hw / cam_fp / "
                                 "is_metric)")
            self._verify_sources()
            self.T, self.N, self.H, self.W = (int(x) for x in a.attrs["TNHW"])
        except Exception:
            self.close()
            raise

    # ---- source files (stored relative to the rig's dir; resolved the same
    # way VDS / ExternalLinks do — a rig moved with its cameras keeps working) ----
    def _cam_file_paths(self):
        d = osp.dirname(osp.abspath(self.path))
        return [osp.normpath(osp.join(d, str(p))) for p in self._h.attrs["cam_files"]]

    def _verify_sources(self):
        a = self._h.attrs
        files = self._cam_file_paths()
        missing = [p for p in files if not osp.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{self.path}: missing camera source file(s) — VDS would "
                f"silently read fillvalues and ExternalLinks raise: {missing}")
        T = int(a["TNHW"][0])                       # the rig's own T
        N = int(a["TNHW"][1])
        hw = [(int(a["cam_hw"][i][0]), int(a["cam_hw"][i][1]))
              for i in range(N)]
        # the sections the rig actually maps (a VDS dataset present in the rig,
        # plus /img when it has image links) — each camera must still hold them
        # with the mapped shape, because a del_section / re-store at a new (h, w)
        # leaves the file size AND THW unchanged and the VDS would silently read
        # the fillvalue.
        has_img = self.has("img")
        vds_secs = [n for n in SECTIONS if n in self._h]
        for i, p in enumerate(files):
            with h5py.File(p, "r") as c:
                thw = c.attrs.get("THW")
                cam = None if thw is None else tuple(int(x) for x in thw)
                Tc, Hc, Wc = (int(x) for x in a["cam_fp"][i])
                if (cam != (Tc, Hc, Wc)
                        or (Tc, Hc, Wc) != (T, hw[i][0], hw[i][1])):
                    raise ValueError(
                        f"{self.path}: camera {i} ({p}) no longer matches the "
                        f"camera THW stored at build time (it changed) — the "
                        f"VDS would misread")
                for name in vds_secs:               # the mapped sections, per camera
                    if name not in c:
                        raise ValueError(
                            f"{self.path}: camera {i} ({p}) no longer has /{name}, "
                            f"which the rig maps — the VDS would read the fillvalue")
                    # the camera stores (T, ...) in every mapped section: (T, H, W)
                    # for the frame sections and (T, 3, 3) / (T, 4, 4) for K / T
                    want = (T, *FRAME_SHAPE[name]) if name in FRAME_SHAPE \
                        else (T, hw[i][0], hw[i][1])
                    if c[name].shape != want:
                        raise ValueError(
                            f"{self.path}: camera {i} ({p}) /{name} is now "
                            f"{c[name].shape} but the rig mapped {want} — the VDS "
                            f"would misread")
                if has_img:
                    ig = c.get("img")
                    if ig is None or any(f"{t}" not in ig for t in range(T)):
                        raise ValueError(
                            f"{self.path}: camera {i} ({p}) no longer has all the "
                            f"/img frames the rig links to — the ExternalLink would raise")

    # ---- properties ----
    @property
    def shape(self):
        return self.T, self.N, self.H, self.W

    def __len__(self):
        return self.T

    @property
    def sections(self):
        return {n for n in ("img", *SECTIONS) if self.has(n)}

    def __repr__(self):
        return (f"<{type(self).__name__} {self.path} shape={self.shape} "
                f"sections={sorted(self.sections)}>")

    @property
    def cam_files(self):
        """List[str] of the source camera files (absolute) this rig references
        (stored relative to the rig file's directory)."""
        return self._cam_file_paths()

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
        /img entry is an ExternalLink to that camera's /img/{t})."""
        self._t(t, n)
        if not self.has("img"):
            raise KeyError("no img frames")
        return decode_rgb(self._h["img"][f"{t}_{n}"][:].tobytes())

    def bgr(self, t=0, n=0):
        """Frame (t, n) decoded as camera n's NATIVE (H, W, 3) uint8 BGR (a
        channel-reversed copy of rgb(t, n) — for direct cv2 use)."""
        return self.rgb(t, n)[:, :, ::-1].copy()

    def depth_mm(self, t=0, n=0):
        """Frame (t, n) depth in MM: (Hmax, Wmax) uint16; pad rows/cols (the
        VDS fillvalue region) read back as 0."""
        return mm_frame(self._f("depth", t, n), self.H, self.W)

    def depth_meters(self, t=0, n=0):
        """Frame (t, n) depth in METERS: (Hmax, Wmax) float32; pad reads back
        0.0."""
        return meters_frame(self._f("depth", t, n), self.H, self.W)

    def depth_conf(self, t=0, n=0):
        """Frame (t, n) confidence: (Hmax, Wmax) f32 0-1; pad reads back 0.0."""
        return conf_frame(self._f("depth_conf", t, n), self.H, self.W)

    def mask(self, t=0, n=0):
        """Frame (t, n) mask: (Hmax, Wmax) bool; the pad region reads False
        (the VDS fillvalue)."""
        return self._f("mask", t, n)

    def K(self, t=0, n=0):
        return self._f("K", t, n)

    def pose_c2w(self, t=0, n=0):
        """Frame (t, n) c2w pose: (4, 4) float32 camera-to-world."""
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
        # (the output dir is created by the base create in step 2 — not here,
        # so a validation failure below leaves no empty directory behind)
        # 1) open + validate everything (no file written yet); a failure
        # opening the k-th camera still closes cameras 0..k-1 deterministically
        cams = []
        try:
            for p in absp:
                cams.append(H5Cam.open(p))
            vals, meta = self._validate(cams)
        finally:
            for c in cams:
                c.close()
        # 2) write to a temp file through a base-class handle (the /meta group
        # and the set_meta serialization are H5Base's), then rename into place;
        # a failed write aborts the temp file
        tmp = f"{out_path}.tmp.{os.getpid()}"
        handle = object.__new__(H5CamRig)          # bypass the rig's __init__
        H5Base.__init__(handle, tmp, create=True)
        try:
            self._write(handle, absp, vals, meta, suffix)
            handle.close()
            os.replace(tmp, out_path)
        except Exception:
            handle._abort_create()                 # close + remove the half-made file
            raise
        self._h = h5py.File(out_path, "r")
        self._created = False

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
        has_rgb = [c.has("img") for c in cams]
        if any(has_rgb) and not all(has_rgb):
            raise ValueError(f"img must be present in ALL cameras or in NONE "
                             f"(got {has_rgb})")
        secs = [tuple(n for n in SECTIONS if n in c._h) for c in cams]
        if len(set(secs)) != 1:
            raise ValueError(f"camera section sets differ: {secs}")
        secs = secs[0]
        dhw = []
        for n, c in enumerate(cams):
            for name in ("depth", "depth_conf"):
                if name in c._h:
                    s = c._h[name].shape[1:]
                    if s != (c.H, c.W):
                        raise ValueError(f"cam {n} {name} is stored at {tuple(s)} "
                                         f"but the camera is ({c.H}, {c.W}): the rig "
                                         f"maps one VDS block per camera and cannot "
                                         f"upscale downsampled depth")
            if has_rgb[n]:                        # size check from the SOF header
                for t in range(c.T):
                    img_ds = c._h["img"][f"{t}"]
                    b = bytes(img_ds[:65536])    # covers a maximal APP1 before SOF
                    hw = jpeg_hw(b)
                    if hw is None:                # no SOF in the prefix: the full read
                        b = bytes(img_ds[:])
                        hw = jpeg_hw(b)
                        if hw is None:
                            hw = decode_rgb(b).shape[:2]
                    if hw != (c.H, c.W):
                        raise ValueError(f"cam {n} img frame {t} size {hw} "
                                         f"!= declared ({c.H},{c.W})")
            dhw.append((c.H, c.W))
        # shared meta: inherited only when EVERY camera agrees (no None mixed in)
        shared = {}
        for k in ("split", "tags"):
            vals = [c.meta(k) for c in cams]
            if all(v is not None for v in vals) and len({repr(v) for v in vals}) == 1:
                shared[k] = vals[0]
        return (T, H, W, N, has_rgb[0], secs, dhw), (dataset, scene, is_metric, shared)

    def _write(self, rig, absp, vals, meta, suffix):
        """Write the rig file (an open H5Base handle on the temp path)."""
        T, H, W, N, has_rgb, secs, dhw = vals
        dataset, scene, is_metric, shared = meta
        h = rig._h
        # references are stored relative to the rig file's directory: VDS and
        # ExternalLinks resolve that way, so a rig copied/moved together with
        # its cameras keeps working (the fingerprint is content-based, not
        # mtime-based, so `cp`/`rsync`/git-checkout don't break it).
        d = osp.dirname(rig.path) or "."
        relp = [osp.relpath(p, d) for p in absp]
        rig.set_attr("TNHW", np.array((T, N, H, W), np.int64))
        rig.set_attr("is_metric", is_metric)
        rig.set_attr("cam_files", np.array(relp, dtype=_STR))
        rig.set_attr("cam_hw", np.array(dhw, np.int64))
        # (T, H, W) per camera — the camera's own THW is the fingerprint (the
        # file SIZE is deliberately NOT stored: a metadata edit on a camera
        # grows the HDF5 file, but a VDS is a live reference, so that is fine)
        fps = [(T, hh, ww) for (hh, ww) in dhw]
        rig.set_attr("cam_fp", np.array(fps, np.int64))
        # /img: nothing stored — each (t, n) is an ExternalLink to the
        # camera's own /img/{t} (variable-length jpg cannot be a VDS).
        # rgb(t, n) decodes at camera n's NATIVE (H, W).
        if has_rgb:
            rg = h.create_group("img")
            for n, r in enumerate(relp):
                for t in range(T):
                    rg[f"{t}_{n}"] = h5py.ExternalLink(r, f"/img/{t}")
        # fixed-shape sections: VDS referencing the cameras
        for name in secs:
            s = SECTIONS[name]
            fixed = not s.frame2d               # (3,3)/(4,4) frames
            shp = FRAME_SHAPE[name] if fixed else (H, W)
            lay = h5py.VirtualLayout(shape=(T, N, *shp), dtype=s.dtype)
            for n, (hh, ww) in enumerate(dhw):
                # ONE mapping per camera: a full-size camera maps the whole block,
                # a padded one the (hh, ww) part (pad -> fillvalue); the fixed
                # K/T map the (3,3)/(4,4) block. One mapping reads back
                # identically for all three cases — no per-frame blocks.
                bh, bw = FRAME_SHAPE[name] if fixed else (hh, ww)
                lay[:, n, :bh, :bw] = h5py.VirtualSource(
                    relp[n], name, shape=(T, bh, bw))
            h.create_virtual_dataset(name, lay, fillvalue=s.fillvalue)
        # /meta: the base set_meta (validated before any write; empty values
        # skipped as before)
        rig.set_meta(**{k: v for k, v in
                        (("dataset", dataset), ("scene", scene),
                         ("suffix", suffix), *shared.items()) if v})
