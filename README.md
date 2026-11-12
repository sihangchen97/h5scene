# h5scene

Per-scene HDF5 writers + file handles, unified layout. Three classes, one
package (`h5scene/`):

| class | file | shape | what it stores |
|---|---|---|---|
| `H5Base` | `h5base.py` | — (base, no shape) | the file handle: open/create, lifecycle, generic `attr`/`set_attr` + `meta`/`set_meta` (checked against the subclass's `ATTR_KEYS`/`META_KEYS`), `write_section` |
| `H5Cam` | `h5cam.py` | (T, H, W) | single-camera sequence: attrs.THW; /rgb/{t} one dataset per frame; /depth /mask /K /T /depth_conf; data stored in the file |
| `H5CamRig` | `h5camrig.py` | (T, N, Hmax, Wmax) | N × H5Cam → one file **of references only (0 bytes of data)**: /rgb via ExternalLinks to each camera's /rgb/{t}; the fixed-shape sections via HDF5 **virtual datasets** |

H5Cam and H5CamRig both inherit H5Base and each define their own
`ATTR_KEYS` / `META_KEYS` (name → expected type, isinstance-checked on
`set_attr`/`set_meta`). The rig has **no `seq`** — it IS the merged time axis:
all cameras must share the same `dataset` and `scene` (checked on build), and
`suffix` is the rig's own variant marker (default empty).

Shared section encode/decode (jpg q≤95 rgb policy, depth mm conversion +
sentinel rules, mask/K/pose broadcast, the `has()` presence map) lives in
`_io.py`; the per-frame decode helpers are rank-agnostic (both classes pass
their own frame: `h['depth'][t]` vs `h['depth'][t, n]`).

## Layout (H5Cam = the scene layout with N collapsed; the rig maps onto it)

```
<scene>.h5
  /rgb/{t}  or  /rgb/{t}_{n}   uint8 1-D image BYTES (jpg q<=95; q>95 or other
                 encodings re-encoded to q95) — one dataset per frame
  /depth         uint16 (T,N,h,w) mm, 0 AND 65535 invalid, lossless gzip4+shuffle [optional]
  /mask          bool   (T,N,H,W) 1 = valid, 0 = padding, gzip4   [optional]
  /K             f32    (T,N,3,3); (3,3) input broadcasts     [optional]
  /T             f32    (T,N,4,4) c2w; (4,4) input broadcasts [optional]
  /depth_conf    f16    (T,N,h,w) 0-1, same (h,w) as /depth, gzip4  [optional]
  /meta/{dataset,scene[,seq],suffix,split,meta_json}  utf-8 [optional]
  #   dataset/scene = the location identity; seq = the camera's sequence (H5Cam
  #   only); (dataset, scene, seq) shared by several h5 = the same scene's
  #   multi-view cameras -> merge them with H5CamRig
  /meta/frame_indices  int64 (T,) source frame numbers       [optional]
  /meta/tags/*         utf-8 (data_type/dynamics/environment/…) [optional]
  attrs: is_metric bool (data-side)
         H5Cam:  THW = int64 (T, H, W)
         H5CamRig: TNHW = int64 (T, N, Hmax, Wmax),
                   cam_files = str[] absolute source cam files,
                   cam_hw = int64 (N, 2) per-camera (H, W)
```

## Usage

```python
from h5scene import H5Base, H5Cam, H5CamRig

# single-camera sequence
cam = H5Cam("cam.h5", T, H, W)
cam.set_rgb(rgb)                    # (T,H,W,3) uint8 / (T,) image-bytes / paths
cam.set_depth(depth_m)              # float meters or uint16 mm (>65.535 m -> 0);
                                    # may be stored at (h,w) < (H,W) if the aspect ratio matches
cam.set_depth_conf(conf)            # same (h,w) as depth
cam.set_mask(mask); cam.set_K(K); cam.set_pose(pose)
cam.set_meta(dataset=…, scene=…, seq=…, suffix=…, tags={…})
cam.close()
with H5Cam("cam.h5") as cam:
    cam.shape; cam.shape == (T, H, W)
    cam.rgb(0)                      # (H, W, 3) uint8
    cam.depth(0, meters=False); cam.K(0); cam.pose(0)
    cam.depth_mask(0, conf_min=…)   # (H,W) bool, derived
    cam.dataset; cam.scene; cam.seq; cam.suffix; cam.tags
    cam.set_attr("is_metric", True); cam.attr("THW")
    cam.set_meta("scene", "x")      # the generic single-key accessors

# N cameras -> one references-only file (0 bytes of data in the rig file)
rig = H5CamRig("rig.h5", ["cam0.h5", "cam1.h5", …], suffix="q")   # cams must share dataset+scene
with H5CamRig("rig.h5") as rig:   # verifies the camera files exist
    rig.shape                                  # (T, N, Hmax, Wmax)
    rig.rgb(t, n)                            # camera n's NATIVE (H, W, 3)
    rig.depth(t, n); rig.depth_mask(t, n)   # (Hmax, Wmax) — VDS; pad rows/cols = 0/False
    rig.cam_files; rig.cam_hw               # sources + per-camera (H, W)
```

Rig rules: same T / is_metric / section set; **H or W identical across all
cameras** (the other axis = max, top-left aligned, pad covered by the VDS
fillvalue); missing a camera file → `FileNotFoundError` on open (a VDS would
otherwise silently read the fillvalue). Camera files stay referenced (absolute
paths in `attrs.cam_files`) — keep them where the rig expects them.

Install editable: `pip install -e .` (needs numpy / opencv-python / h5py).
Test: `pixi run -e test test`.
