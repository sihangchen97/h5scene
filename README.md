# h5scene

Per-scene HDF5 writers + file handles, unified layout. Three classes, one
package (`h5scene/`):

| class | file | shape | what it stores |
|---|---|---|---|
| `H5Base` | `h5base.py` | — (base, no shape) | the file handle: open/create, lifecycle, generic `attr`/`set_attr` + `meta`/`set_meta` (checked against the subclass's `ATTR_KEYS`/`META_KEYS`), `write_section` |
| `H5Cam` | `h5cam.py` | (T, H, W) | single-camera sequence: attrs.THW; /img/{t} one dataset per frame; /depth /mask /K /T /depth_conf; data stored in the file |
| `H5CamRig` | `h5camrig.py` | (T, N, Hmax, Wmax) | N × H5Cam → one file **of references only (0 bytes of data)**: /img via ExternalLinks to each camera's /img/{t}; the fixed-shape sections via HDF5 **virtual datasets** |

H5Cam and H5CamRig both inherit H5Base and each define their own
`ATTR_KEYS` / `META_KEYS` (name → expected type, isinstance-checked on
`set_attr`/`set_meta`). The rig has **no `seq`** — it IS the merged time axis:
all cameras must share the same `dataset` and `scene` (checked on build), and
`suffix` is the rig's own variant marker (default empty).

Shared section encode/decode (jpg q≤95 img policy, depth mm conversion +
sentinel rules, mask/K/pose broadcast, the `has()` presence map) lives in
`codecs/`; the per-frame decode helpers are rank-agnostic (both classes pass
their own frame: `h['depth'][t]` vs `h['depth'][t, n]`).

## Layout (H5Cam = the scene layout with N collapsed; the rig maps onto it)

```
<scene>.h5
  /img/{t}  or  /img/{t}_{n}   uint8 1-D image BYTES (jpg q<=95; q>95 or other
                 encodings re-encoded to q95) — one dataset per frame
  /depth         uint16 (T,N,h,w) mm, 0 AND 65535 invalid, lossless gzip4+shuffle [optional]
  /mask          bool   (T,N,H,W) 1 = valid, 0 = padding, gzip4   [optional]
  /K             f32    (T,N,3,3); (3,3) input broadcasts     [optional]
  /T             f32    (T,N,4,4) c2w; (4,4) input broadcasts [optional]
  /depth_conf    f16    (T,N,h,w) 0-1, same (h,w) as /depth, gzip4  [optional]
  /meta/{dataset,scene[,seq],suffix,split}  utf-8 [optional]
  #   dataset/scene = the location identity; seq = the camera's sequence (H5Cam
  #   only); (dataset, scene, seq) shared by several h5 = the same scene's
  #   multi-view cameras -> merge them with H5CamRig
  /meta/src_frame_idxs  int64 (T,) source frame numbers       [optional]
  /meta/src_frame_ids   utf-8 (T,) STRING frame ids (tum / dl3dv) [optional]
  /meta/timestamps      f64 (T,) frame timestamps             [optional]
  /meta/tags/*         utf-8 (data_type/dynamics/environment/…) [optional]
  attrs: file_type str ("h5cam" / "h5camrig") — pinned at create/build
         is_metric bool (data-side)
         H5Cam:  THW = int64 (T, H, W)
         H5CamRig: TNHW = int64 (T, N, Hmax, Wmax),
                   cam_files = str[] source cam files relative to the rig's
                   directory (VDS/ExternalLinks resolve them the same way),
                   cam_hw = int64 (N, 2) per-camera (H, W)
                   (open re-reads each camera's THW and checks (T, *cam_hw[n])
                   plus every mapped section's presence + shape)
```

## Usage

```python
from h5scene import H5Base, H5Cam, H5CamRig

# single-camera sequence
cam = H5Cam.create("cam.h5", T, H, W)   # create (overwrites); is_metric defaults False
# the writers are SPLIT BY INPUT KIND (the colour order on disk is unknowable):
cam.set_rgb(rgb)                    # uint8 RGB ARRAYS only: (T,H,W,3) / (H,W,3) (T=1)
cam.set_rgb_bytes(imgs)             # image BYTES (T,): q<=95 cv2 jpgs kept verbatim, else re-encoded q95
cam.set_rgb_paths(paths)            # file PATHS (T,)
cam.set_bgr(bgr)                    # (T,H,W,3) / (H,W,3) uint8 BGR ARRAY ONLY (no bytes/paths)
cam.set_depth_mm(d_mm)              # INTEGER mm ONLY (a float means meters -> use set_depth_meters)
cam.set_depth_meters(d_m)           # float METERS ONLY (an integer means mm -> use set_depth_mm)
                                    # may be stored at (h,w) < (H,W) if the aspect ratio matches
cam.set_depth_conf_01(conf)        # 0-1 FLOAT ONLY (integers rejected); same (h,w) as depth
cam.set_mask(mask)
cam.set_K(K); cam.set_pose_c2w(pose_c2w)   # conventions checked: K pinhole, pose c2w
cam.set_meta(dataset=…, scene=…, seq=…, suffix=…, src_frame_ids=[str…],
             timestamps=[float…], tags={…})   # multi-key (set the keys given)
# per-frame / streaming write (peak memory = one frame):
cam.allocate(sections=("img", "depth", "K", "T"))    # pre-create the datasets
for t in range(T):
    cam.write_frame(t, rgb=rgb[t], depth_mm=d[t], K=K, pose_c2w=P)
cam.finalize()                        # checks every t was written
cam.set_src_frame_ids(["1341846313.553992", …])     # index_of(fid) / indices_of(…) O(1)
cam.close()
with H5Cam.open("cam.h5") as cam:   # open: read-only by default (mode="r" or "r+")
    cam.shape; cam.shape == (T, H, W)
    cam.rgb(0) / cam.bgr(0)        # (H, W, 3) uint8 RGB / BGR
    cam.img_bytes(0)                # the stored jpg BYTES, undecoded
    cam.depth_mm(0); cam.depth_meters(0); cam.K(0); cam.pose_c2w(0)
    cam.depth_mask(0, conf_min=…)   # (H,W) bool, derived
    cam.dataset; cam.scene; cam.seq; cam.suffix; cam.tags
    cam.attr("THW"); cam.has("img")
H5Cam.peek("cam.h5")                 # the listing fields WITHOUT the full open validation
# CLI: python -m h5scene.verify <file | dir> [--json out.json]   # the post-conversion
# check (no raw 65535, K/pose conventions, every img frame decodes at (H, W)); exit 1 on FAIL
with H5Cam.open("cam.h5", "r+") as cam:   # writable: the generic single-key accessors
    cam.set_attr("is_metric", True)
    cam.set_meta("scene", "x")            # single-key form: value = set, None = remove

# N cameras -> one references-only file (0 bytes of data in the rig file)
rig = H5CamRig.build("rig.h5", ["cam0.h5", "cam1.h5", …], suffix="q")
#   out path must not be one of the cameras; the output dir is created
with H5CamRig.open("rig.h5") as rig:   # verifies the cameras still match
                                       # (T, *cam_hw[n]) + mapped sections
    rig.shape                                  # (T, N, Hmax, Wmax)
    rig.rgb(t, n) / rig.bgr(t, n)          # camera n's NATIVE (H, W, 3) RGB / BGR
    rig.depth_mm(t, n); rig.depth_meters(t, n); rig.depth_mask(t, n)
                                           # (Hmax, Wmax) — VDS; pad rows/cols = 0/False
    rig.cam_files; rig.cam_hw               # sources + per-camera (H, W)
```

Rig rules (all checked at BUILD, before any file is written; the output
directory is created, and the build is write-temp-then-rename, so a failure
never touches an existing output): same T / is_metric / **dataset / scene**
(the cameras' seqs merge into one time axis — the rig has none); img present
in ALL cameras or in NONE; same section set; depth at the camera's NATIVE
(H, W) (downsampled depth is rejected — a VDS block cannot upscale);
**H or W identical across all cameras** (the other axis = max, top-left
aligned, pad covered by the VDS fillvalue); the output path is not one of
the cameras (it would be overwritten). Shared split / tags are inherited when
every camera agrees.
A missing or REWRITTEN camera file → `FileNotFoundError` / `ValueError` on
open (a VDS would otherwise silently read the fillvalue). Camera files stay
referenced (paths relative to the rig's directory in `attrs.cam_files` —
copy or move the rig together with its cameras and it keeps working).

Install editable: `pip install -e .` (needs numpy / opencv-python / h5py).
Test: `pixi run -e test test`.
