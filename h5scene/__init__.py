"""h5scene: per-scene HDF5 writers + file handles, unified layout.

Three classes (H5Cam and H5CamRig both inherit H5Base):

- H5Base    (h5base.py)   — the file handle itself (no in-memory data, no
  shape): open/create + lifecycle, the generic attr(name) / set_attr(name,
  value) + meta(name) / set_meta(name, value) accessors (checked against the
  subclass's ATTR_KEYS / META_KEYS registries, isinstance-validated), and the
  one low-level write_section.
- H5Cam     (h5cam.py)    — single-camera sequence (T, H, W): the scene
  layout below with the N axis collapsed; attrs.THW; /rgb/{t} one dataset per
  frame; attrs.is_metric defaults False; each set_*/read is a (T, ...) method.
- H5CamRig  (h5camrig.py) — N H5Cam files -> one multi-camera (T, N, Hmax,
  Wmax) file maintained as REFERENCES only (0 bytes of data in the rig file):
  /rgb via ExternalLinks to each camera's /rgb/{t}; the fixed-shape sections
  via HDF5 virtual datasets. ALL cameras must share the SAME dataset and scene
  (checked on build; the rig has no seq of its own — it IS the merged time
  axis); H or W identical across cameras (the other = max, top-left aligned,
  pad covered by the VDS fillvalue). rgb(t, n) decodes at camera n's NATIVE
  (H, W); the VDS sections at the common (Hmax, Wmax).

File layout (H5Cam = the scene layout with N collapsed; H5CamRig maps the N
axis onto it — written "[N]" where the rig adds one axis):

    <scene>.h5
      /rgb/{t} (cam) | /rgb/{t}_{n} (rig)   uint8 1-D image BYTES (jpg q<=95;
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
      /meta/{dataset,scene[,seq],suffix,split,meta_json}  utf-8 (dataset-side
              context; meta_json = source verbatim).
              The location identity: (dataset, scene[, seq]) shared by several
              h5 = the same scene's cameras; seq is per-camera (the rig has
              none); suffix marks variants — see docs/meta_key_decisions.md
      /meta/frame_indices  int64 (T,) source frame numbers        [optional]
      /meta/tags/*         utf-8 strings (data_type/dynamics/...)  [optional]
      attrs: is_metric bool (data-side)
             H5Cam:  THW = int64 (T, H, W)
             H5CamRig: TNHW = int64 (T, N, Hmax, Wmax),
                       cam_files = str[] absolute source cam files,
                       cam_hw = int64 (N, 2) per-camera (H, W)

H5Cam usage:

    cam = H5Cam(path, T, H, W, is_metric=False)   # create (overwrites)
    cam.set_rgb(rgb)          # (T,H,W,3) uint8 / flat (T,) image bytes / paths
    cam.set_depth(d); cam.set_depth_conf(c); cam.set_mask(m)
    cam.set_K(K); cam.set_pose(P)
    cam.set_meta(dataset=..., scene=..., seq=..., suffix=..., split=...,
                 meta_json=..., frame_indices=..., tags={...})   # multi-key
    cam.close()                          # (or: with H5Cam(...) as cam: ...)

    cam = H5Cam(path)                    # open an existing file (THW from attrs)
    cam.shape, cam.is_metric, cam.has(name)
    cam.rgb(t)                           # decoded (H, W, 3) uint8 (no bytes reader)
    cam.depth(t, meters=...) / cam.K(t) / cam.pose(t) / cam.mask(t) / cam.depth_conf(t)
    cam.depth_mask(t, conf_min=...)
    cam.dataset, cam.scene, cam.seq, cam.suffix, cam.split, cam.meta_json
    cam.frame_indices, cam.tags
    cam.set_attr("is_metric", True); cam.attr("THW")     # the base accessors
    H5Base.set_meta(cam, "scene", "x")                   # the single-key base form

H5CamRig usage:

    rig = H5CamRig(path, cam_paths, suffix=...)   # cam_paths: existing .h5 files
    rig = H5CamRig(path)                        # open (checks cam_files exist)
    rig.shape                                   # (T, N, Hmax, Wmax)
    rig.rgb(t, n)   rig.depth(t, n, meters=...)  # (Hmax, Wmax); pad = 0/False
    rig.cam_files, rig.cam_hw                   # the absolute source files
"""
from .h5base import H5Base
from .h5cam import H5Cam
from .h5camrig import H5CamRig

__all__ = ["H5Base", "H5Cam", "H5CamRig"]
