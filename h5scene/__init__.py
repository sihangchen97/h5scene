"""h5scene: per-scene HDF5 writers + file handles, unified layout.

Three classes (H5Cam and H5CamRig both inherit H5Base):

- H5Base    (h5base.py)   — the file handle itself (no in-memory data, no
  shape): open/create + lifecycle, the generic attr(name) / set_attr(name,
  value) + meta(name) / set_meta(name, value) accessors (checked against the
  subclass's ATTR_KEYS / META_KEYS registries, isinstance-validated), and the
  one low-level write_section.
- H5Cam     (h5cam.py)    — single-camera sequence (T, H, W): the scene
  layout below with the N axis collapsed; attrs.THW; /img/{t} one dataset per
  frame; attrs.is_metric defaults False; each set_*/read is a (T, ...) method.
- H5CamRig  (h5camrig.py) — N H5Cam files -> one multi-camera (T, N, Hmax,
  Wmax) file maintained as REFERENCES only (0 bytes of data in the rig file):
  /img via ExternalLinks to each camera's /img/{t}; the fixed-shape sections
  via HDF5 virtual datasets. ALL cameras must share the SAME dataset and scene
  (checked on build; the rig has no seq of its own — it IS the merged time
  axis); H or W identical across cameras (the other = max, top-left aligned,
  pad covered by the VDS fillvalue). rgb(t, n) decodes at camera n's NATIVE
  (H, W); the VDS sections at the common (Hmax, Wmax). attrs: TNHW, is_metric,
  cam_files (relative to the rig file), cam_hw, cam_fp (per-camera content
  fingerprint, verified on open).

File layout (H5Cam = the scene layout with N collapsed; H5CamRig maps the N
axis onto it — written "[N]" where the rig adds one axis):

    <scene>.h5
      /img/{t} (cam) | /img/{t}_{n} (rig)   uint8 1-D image BYTES (jpg q<=95;
                  q>95 or other encodings re-encoded to q95) — one dataset
                  per frame (T in a cam, T*N in a rig)
      /depth        uint16 (T[,N,]h,w) mm, 0 AND 65535 both invalid, gzip4
                    (+shuffle in a cam), lossless; h<=H w<=W, aspect ratio
                    H/W matches; read resizes (h,w) -> (H,W) (nearest)  [optional]
      /mask         bool   (T[,N,]H,W) 1 = valid, 0 = padding, gzip4  [optional]
      /K            f32    (T[,N,]3,3); (3,3) input broadcasts     [optional]
      /T            f32    (T[,N,]4,4) c2w; (4,4) input broadcasts [optional]
      /depth_conf   f16    (T[,N,]h,w) 0-1, same (h,w) as /depth, gzip4  [optional]
      derived: .depth_mask(t[, n], conf_min=...) — (H,W) bool, not stored:
                 depth not an invalid sentinel (0 AND 65535 excluded) and,
                 if depth_conf present, conf > (0 if conf_min is None, else conf_min)
      /meta/{dataset,scene[,seq],suffix,split}  utf-8 (dataset-side context).
              The location identity: (dataset, scene[, seq]) shared by several
              h5 = the same scene's cameras; seq is per-camera (the rig has
              none); suffix marks variants — see docs/schema.md
      /meta/src_frame_idxs  int64 (T,) source frame numbers        [optional]
      /meta/tags/*         utf-8 strings (data_type/dynamics/...)  [optional]
      attrs: is_metric bool (data-side)
             H5Cam:  THW = int64 (T, H, W)
             H5CamRig: TNHW = int64 (T, N, Hmax, Wmax),
                       cam_files = str[] source cam files relative to the
                       rig file's directory (the VDS/ExternalLinks resolve
                       them the same way — a rig moved with its cameras works),
                       cam_hw = int64 (N, 2) per-camera (H, W),
                       cam_fp = int64 (N, 4) per-camera (T, H, W, size)
                       content fingerprint (verified at open; legacy (N,5)
                       rows with mtime_ns are still accepted)

H5Cam usage:

    cam = H5Cam.create(path, T, H, W, is_metric=False)   # create (overwrites)
    cam.set_rgb(rgb)          # (T,H,W,3) uint8 / (H,W,3) (T=1) / (T,) image bytes / paths
    cam.set_bgr(bgr)         # (T,H,W,3) / (H,W,3) uint8 BGR ARRAY ONLY (arrays only)
    cam.set_depth_mm(d_mm); cam.set_depth_meters(d_m)     # unit is in the name
    cam.set_depth_conf_01(c) # (T,h,w) 0-1 float ONLY
    cam.set_mask(m)
    cam.set_K(K); cam.set_pose_c2w(P)
    cam.set_meta(dataset=..., scene=..., seq=..., suffix=..., split=...,
                 src_frame_idxs=..., tags={...})   # multi-key
    cam.close()                          # (or: with H5Cam.create(...) as cam: ...)

    cam = H5Cam.open(path)               # open an existing file ("r" / "r+")
    cam.shape, cam.is_metric, cam.has(name), cam.sections
    cam.rgb(t) / cam.bgr(t)             # decoded (H, W, 3) uint8 RGB / BGR (no bytes reader)
    cam.depth_mm(t) / cam.depth_meters(t) / cam.K(t) / cam.pose_c2w(t) / cam.mask(t) / cam.depth_conf(t)
    cam.depth_mask(t, conf_min=...)
    cam.dataset, cam.scene, cam.seq, cam.suffix, cam.split
    cam.src_frame_idxs, cam.tags
    cam.set_attr("is_metric", True); cam.attr("THW")     # the base accessors
    cam.set_meta("scene", "x")                          # single-key form (None = remove)

H5CamRig usage:

    rig = H5CamRig.build(path, cam_paths, suffix=...)   # validate (out path not
                                                        # a camera), mkdir, temp, rename
    rig = H5CamRig.open(path)                           # verifies the cam content fingerprint
    rig.shape                                   # (T, N, Hmax, Wmax)
    rig.rgb(t, n) / rig.bgr(t, n)   rig.depth_mm(t, n) / rig.depth_meters(t, n)
                                             # (Hmax, Wmax); pad = 0/False
    rig.cam_files, rig.cam_hw                   # the source files (absolute) + (H, W)
"""
from .h5base import H5Base, MetaSpec
from .h5cam import H5Cam
from .h5camrig import H5CamRig

__all__ = ["H5Base", "H5Cam", "H5CamRig", "MetaSpec"]
