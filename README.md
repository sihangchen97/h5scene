# h5scene

Per-scene HDF5 writers + file handles with the unified `(T, N, H, W)` layout
(T = time, N = camera). Three classes, one package:

| class | file | shape | what it stores |
|---|---|---|---|
| `H5Base` | `h5base.py` | (T, N, H, W) | generic writer + read handle, layout below, data stored in the file |
| `H5Cam`  | `h5cam.py`  | (T, H, W)    | single-camera sequence: same layout, N collapsed, `attrs.THW`, `/rgb/{t}` |
| `H5CamRig` | `h5camrig.py` | (T, N, Hmax, Wmax) | N x H5Cam -> one file **of references only (0 bytes of data)**: `/rgb` via ExternalLinks to each camera's `/rgb/{t}`, `/depth` `/mask` `/K` `/T` `/depth_conf` via HDF5 **virtual datasets** |

All three are plain file handles — no in-memory data. The rig file is also a
valid `H5Base` file (open it with `H5Base` directly); `H5CamRig(path)` adds
a source-file guard (missing/renamed camera files would otherwise make the
VDS reads silently return fillvalues).

## Layout (H5Base; H5Cam = same with N collapsed)

```
<scene>.h5
  /rgb/{t}_{n}   uint8 1-D image BYTES (jpg q<=95; q>95 or other encodings
                 re-encoded to q95) — key derived from (t, n)
  /depth         uint16 (T,N,H,W) mm, 0 AND 65535 invalid, lossless gzip4+shuffle [optional]
  /mask          bool   (T,N,H,W) 1 = valid, 0 = padding, gzip4   [optional]
  /K             f32    (T,N,3,3); (3,3) input broadcasts     [optional]
  /T             f32    (T,N,4,4) c2w; (4,4) input broadcasts [optional]
  /depth_conf    f16    (T,N,H,W) 0-1, gzip4                  [optional]
  /meta/{dataset,scene,seq,suffix,split,meta_json}  utf-8 [optional]
  #   4 location keys: (dataset, scene[, seq]) shared by several h5 = the
  #   same scene's multi-view cameras; suffix marks variants (train/val, camera)
  /meta/frame_indices  int64 (T,) — only when != arange(T)    [optional]
  /meta/tags/*         utf-8 (data_type/dynamics/environment/…) [optional]
  attrs: TNHW = int64 (T, N, H, W), is_metric bool (data-side)
```

## Usage

```python
from h5scene import H5Base, H5Cam, H5CamRig

# generic (T, N, H, W) file
s = H5Base("s.h5", T, N, H, W)
s.set_rgb(rgb)                    # (T,N,H,W,3) uint8 / (H,W,3) / image-bytes seq / file paths
s.set_depth(depth_m)              # float meters or uint16 mm (>65.535 m -> 0);
                                  # may be stored at (h,w) < (H,W) if aspect ratio matches
s.set_depth_conf(conf)            # must be the same (h,w) as depth; -> (H,W) on read
s.set_mask(mask)                  # full (H,W) only: (T,N,H,W) / (H,W) broadcast / (T,N) slots
s.set_K(K); s.set_pose(pose)
s.set_meta(dataset=…, scene=…, seq=…, suffix=…, split=…, meta_json=…,
           frame_indices=[…], tags={"data_type": "real", …})   # see docs/meta_key_decisions.md
s.close()

with H5Base("s.h5") as s:        # open
    s.shape, s.keys, s.has("depth")
    s.rgb(t, n)                  # decoded (H,W,3) uint8 (stored jpg q95, no bytes reader)
    s.depth(t, n, meters=True); s.K(t, n); s.pose(t, n)
    # s.depth / s.depth_conf come back at (H, W) — smaller stored sections are
    # resized (INTER_NEAREST) on read
    s.mask(t, n); s.depth_conf(t, n)
    s.dataset; s.scene; s.seq; s.suffix; s.split; s.meta_json
    s.is_metric; s.frame_indices; s.tags
    s.depth_mask(t, n, conf_min=…)    # (H,W) bool, derived: depth valid (0/65535 excluded)
                                       # and, if depth_conf present, conf > 0 (or > conf_min)

# single-camera sequence
cam = H5Cam("cam.h5", T, H, W)
cam.set_rgb(per_frame_bytes); cam.set_depth(depth_m); cam.close()
cam = H5Cam("cam.h5"); cam.rgb(0)   # (H, W, 3)

# N cameras -> one references-only file (0 bytes of data in the rig file)
rig = H5CamRig("rig.h5", ["cam0.h5", "cam1.h5", …], meta={"dataset": "…", "scene": "…"})
with H5CamRig("rig.h5") as rig:     # verifies the camera files exist
    rig.shape                                  # (T, N, Hmax, Wmax)
    rig.rgb(t, n)                            # (h_n, w_n, 3) — the CAMERA'S native size
    rig.depth(t, n); rig.depth_mask(t, n)   # (Hmax, Wmax) — VDS, pad rows/cols = 0/False
    rig.cam_files; rig.cam_hw               # sources + per-camera (H, W)
```

Rig rules: same T / is_metric / section set; **H or W identical across all
cameras** (the other axis = max, top-left aligned, pad covered by the VDS
fillvalue); missing a camera file -> `FileNotFoundError` on open. The camera
files stay referenced (absolute paths in `attrs.cam_files`) — keep them where
the rig expects them. See `docs/meta_key_decisions.md`.

Install editable: `pip install -e .` (needs numpy / opencv-python / h5py).
Test: `pytest` or `python -m pytest tests/`.
