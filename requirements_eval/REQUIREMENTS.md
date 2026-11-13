# Requirements: h5scene for evaluation — from vggt_research (2026-11-13)

Requester: vggt_research eval (sihang). Implementer: h5scene owner.
Plans this serves (canonical, in vggt_research): `eval/PLAN_eval_migration.md`,
`eval/PLAN_eval_datasets.md`, `eval/PLAN_eval_library_tasks.md` (§1 H-1…H-8).
Training-side requests (groups CSR, old-flat compat read, …) live in
vggt_research `dataset/PLAN_h5scene_migration.md` §5 and are **not** repeated here.

## Decisions already made (do not reopen here)

| # | Decision | Effect on h5scene |
|---|---|---|
| D1 | Eval reads the **h5scene schema only** (h5scene = sole reader) | Every eval dataset becomes an `H5Cam` file |
| D2 | Old flat eval data is **re-converted first**, before eval migration Phase 1 | Old-flat compat read is **not** needed for eval |
| D3 | Images are **q95 jpg everywhere**, eval included | `img_policy="passthrough"` is **not** needed for eval |
| D4 | Depth stays uint16 mm; >65.535 m → 0 | Already the h5scene rule ✓ |

## Facts the requests rest on (measured 2026-11-13)

- **Frame ids are strings, not ints**, in most datasets (old flat `/meta/images`, one file per dataset scanned):

  | dataset | example fid | int-like? |
  |---|---|---|
  | tum, bonn | `1341846313.553992` (timestamp) | no |
  | dl3dv | `001dcc…/images_8/frame_00001.png` | no |
  | nrgbd | `img0` | no |
  | sintel | `frame_0001` | no |
  | nuscenes, waymo | `00000_0`, `00000_1` | no |
  | artemis_dfa | `0000_00` | no |
  | 7scenes, scannet, co3d, eth3d, … | `000000` | yes |

  Published frame lists (π³ seq-id-maps, CUT3R/MonST3R slices) key on these names.
  `/meta/src_frame_idxs` (int64) cannot hold them.
- **Per-frame image size varies** inside one scene: co3d (`depth/000001` 696×384,
  `depth/000012` 703×384, …) and waymo. `attrs.THW` is a single (H, W).
- `H5Cam.open` on SpatialBench's 129 files: 0.91 s total (7 ms/file) vs 0.39 s raw h5py.
  Fine today; matters for scene listing over ~5000 scenes on network storage.
- SpatialBench files have `split` unset; vggt_research eval filters scenes by split.

## Requests

Priority: **P0** blocks eval re-conversion (D2); **P1** blocks specific datasets; **P2** convenience.

### E1 (P0) — string frame ids

- **Need:** keep each frame's source name; look frames up by name.
- **Proposed API** (names are a suggestion; owner decides):
  - `cam.set_fids(list[str])` → `/meta/fids`, utf-8, shape (T,).
  - `cam.fids` → `list[str]` (or `None` if absent).
  - `cam.index_of(fid) -> int` and `cam.indices_of(list[str]) -> list[int]`; unknown fid raises `KeyError`.
- **Rules:** length must equal T; fids unique (else `ValueError`); row order = time order;
  can coexist with `src_frame_idxs`; not inherited by `H5CamRig` (same as `src_frame_idxs`).
- **Acceptance:**
  - round-trip of the tum timestamp fids and the dl3dv path fids (see `samples/`) is exact;
  - duplicate or wrong-length fids raise;
  - `index_of` is O(1) after the first call (eval looks up ~10–300 fids per scene).
- Docs: `docs/schema.md`, `README.md`, `__init__.py` docstring.

### E2 (P1) — variable per-frame resolution

- **Need:** co3d (and waymo) frames differ in size within a scene.
- **Two options (owner picks):**
  - **A — pad to max (recommended):** `attrs.THW = (T, Hmax, Wmax)` plus `/frame_hw`
    int64 (T, 2). depth / depth_conf / mask are stored (T, Hmax, Wmax), top-left
    aligned; pad = 0 (depth invalid sentinel) and mask = False (the existing
    "0 = padding" meaning). `rgb(t)`, `depth_mm(t)`, `mask(t)` return the frame's
    native (h_t, w_t). It reuses the rig's existing pad rules. co3d's waste is under 1 %.
  - **B — per-frame datasets:** `/depth/{t}` like `/img/{t}`. More flexible, but needs
    a second code path in every reader and in the rig VDS.
- **Rules:**
  - K(t) refers to frame t's native size.
  - depth must be stored at native size per frame (no smaller (h, w) with variable frames).
  - `H5CamRig.build` should reject variable-size cameras until defined.
- **Acceptance:** the co3d sample (6 frames, heights 696/698/700/703) converts, and each
  frame reads back at its own size with depth bit-identical to the source.

### E3 (P1) — frame-level camera checks and convention text

- **Need:** eval assumes OpenCV pinhole: K pixel-centre convention, `K[2,2] = 1`,
  pose = c2w with x right, y down, z forward. vggt_research pins `K[2,2] = 1`
  (`tests/test_unified_k_convention.py`). h5scene stores K/T raw, with no check.
- **Ask:**
  1. Write the convention in `docs/schema.md`: "pixel (u, v) centre is at coordinate
     (u, v); the image spans [-0.5, W-0.5]". SpatialBench `DATASET.md` says
     "主点=左上像素角" (principal point = top-left pixel corner), which reads like
     the opposite convention, so align the wording.
  2. `set_K` rejects K with `K[2,2] != 1`, non-zero `K[1,0]` / `K[2,0]` / `K[2,1]`,
     or fx/fy ≤ 0. `set_pose_c2w` rejects a last row ≠ [0, 0, 0, 1], or
     |det(R) − 1| > 1e-3.
- **Acceptance:** each bad input raises; current tests still pass.

### E4 (P1) — encoded-bytes reader

- **Need:** eval records an input hash in result provenance, and some adapters take encoded bytes.
- **Ask:** `cam.img_bytes(t) -> bytes` (the stored jpg, undecoded). For the rig it
  follows the ExternalLink.
- **Acceptance:** `cv2.imdecode(img_bytes(t))` (as RGB) equals `rgb(t)`.

### E5 (P1) — per-frame / streaming write

- **Need:** long eval sequences (TUM / ScanNet full length, ≥1000 frames, for drift).
  Today depth must be passed as one (T, H, W) array. Same need as training
  (`dataset/PLAN_h5scene_migration.md` §5.4: artemis 25020 frames).
- **Proposed API:**
  - `cam.allocate(sections=("depth", "K", "T", …), depth_hw=(h, w))` pre-creates datasets with chunks (1, h, w);
  - `cam.write_frame(t, rgb=…, depth_mm=… | depth_meters=…, depth_conf=…, mask=…, K=…, pose_c2w=…)`;
  - `cam.finalize()` checks every t was written for every allocated section, else raises.
  - Encoding rules stay the same as `set_*`.
- **Acceptance:**
  - a 1000-frame synthetic scene written frame-by-frame equals the `set_*` version byte-for-byte;
  - peak memory stays O(one frame);
  - a skipped t makes `finalize` raise.

### E6 (P2) — `verify` command

- **Need:** one check run after every eval conversion (eval plan W5.2 / RULES R1 there).
- **Ask:** `python -m h5scene.verify <file.h5 | dir> [--json out.json]`. Per file it reports:
  - opens OK;
  - `(raw depth == 65535).sum()` (must be 0);
  - valid-depth min/max;
  - E3 K/pose checks;
  - every img decodes at the declared size;
  - fids unique;
  - exit code 1 on any failure.
- Optional: a library function `check_geometry(cam, pairs)` (reprojection consistency),
  so SpatialBench `tests/check_h5_geo.py` can call it instead of keeping its own copy.

### E7 (P2) — light metadata read

- **Ask:** `H5Cam.peek(path) -> dict` with THW, is_metric, sections, dataset, scene, seq,
  split and fids. No section validation, so scene listing stays fast.
- **Acceptance:** 5000 files in under 10 s on local disk.

### E8 (P2) — float meta arrays

- **Ask:** let `MetaSpec` allow float64 (T,) arrays, e.g. `timestamps`.
  `MetaSpec` already checks dtype and length.

## Not requested by eval (on purpose)

- groups CSR: eval frame lists live as JSON in the eval repo;
- `img_policy="passthrough"` (D3);
- old-flat compat read (D2);
- any protocol, metric or adapter code.

## Contents of this folder

- `samples/` — cut-downs of real old-flat files (byte-faithful for the kept frames):
  - `7scenes_pumpkin_seq-01_cut5.h5` (fixed size, int fids);
  - `tum_walking_xyz_cut5.h5` (timestamp fids → E1);
  - `co3d_car_564_cut6.h5`: **incomplete**. The cut script failed on unsorted frame
    order, so regenerate it or ignore it; it is the E2 test input.
- `reference/` — copies of the current eval consumers (`vggt_research_eval_gt.py`,
  `vggt_research_eval_scenes.py`) and the old flat layout spec
  (`vggt_research_preprocess_README.md` §0.1). Read-only context.
- `samples/build_samples.py` — the requester's script that made the cuts; not library code.
