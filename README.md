# h5scene

Per-scene HDF5 writer + `H5Scene` file handle with the unified
`(T, N, …)` layout (T = time, N = camera). One file, two APIs:

- `write_scene_h5(path, rgb, …)` / `verify_scene_h5(path, rgb, …)` — one-shot
  write / byte-level verify (the functional API).
- `H5Scene` — a **direct file handle, no in-memory data**:
  `H5Scene(path, T, N, H, W)` creates (TNHW pinned in `attrs`),
  `H5Scene(path)` opens; every `set_*` validates its shape against the pinned
  TNHW and writes straight into the file; every read comes straight out.

## Layout

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
from h5scene import H5Scene

s = H5Scene("s.h5", T, N, H, W)
s.set_rgb(rgb)                    # (T,N,H,W,3) uint8 / (H,W,3) / image-bytes seq / file paths
s.set_depth(depth_m)              # float meters or uint16 mm (>65.535 m -> 0);
                                  # may be stored at (h,w) < (H,W) if aspect ratio matches
s.set_depth_conf(conf)            # must be the same (h,w) as depth; -> (H,W) on read
s.set_mask(mask)                  # full (H,W) only: (T,N,H,W) / (H,W) broadcast / (T,N) slots
s.set_K(K); s.set_pose(pose)
s.set_meta(dataset=…, scene=…, seq=…, suffix=…, split=…, meta_json=…,
           frame_indices=[…], tags={"data_type": "real", …})   # see docs/meta_key_decisions.md
s.close()

with H5Scene("s.h5") as s:       # open
    s.shape, s.keys, s.has("depth")
    s.rgb(t, n)                  # decoded (H,W,3) uint8 (stored jpg q95, no bytes reader)
    s.depth(t, n, meters=True); s.K(t, n); s.pose(t, n)
    # s.depth / s.depth_conf come back at (H, W) — smaller stored sections are
    # resized (INTER_NEAREST) on read
    s.mask(t, n); s.depth_conf(t, n)
    s.dataset; s.scene; s.seq; s.suffix; s.split; s.meta_json
    s.is_metric; s.frame_indices; s.tags        # tags: {data_type, dynamics, environment, …}
    s.depth_mask(t, n, conf_min=…)    # (H,W) bool, derived: depth valid (0/65535 excluded)
                                       # and, if depth_conf present, conf > 0 (or > conf_min)
```

Install editable: `pip install -e .` (needs numpy / opencv-python / h5py).
Test: `pytest` or `python -m pytest tests/`.
