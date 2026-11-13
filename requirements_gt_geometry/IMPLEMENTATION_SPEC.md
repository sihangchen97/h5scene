# Implementation spec: `H5Geo` — general static geometry, Houdini-style

Response to `REQUIREMENTS_GT_GEOMETRY.md` (2026-11-13). This is a plan only; no code has been written.

**Decided (owner):**
- Class `H5Geo`.
- Houdini element levels and **Houdini attribute names**.
- A full attribute structure.
- `is_metric` defaults to False.
- Colours are stored as uint8.
- Value-range checks run at write and read time, not at open.
- No PLY reader in this library.

All decisions are now made (§12). Compression is settled by the agent experiment in §5.

**Delivery:**
- **Phase 1** (what SpatialBench needs now): P / Cd / N on points, plus triangle topology.
- **Phase 2:** the rest of the schema.

The file schema is fixed in full from Phase 1, so Phase 1 files stay valid.

---

## 0. Review of the request (findings)

| # | Finding | Resolution |
|---|---|---|
| R1 | The request treats point clouds and meshes as one class with optional faces. | Kept. Points with no prims form a point cloud. |
| R2 | The request uses "vertices" to mean positions. In Houdini a **vertex** is a face corner that references a point. | Houdini naming throughout. A vertex is always a corner. |
| R3 | `SECTIONS` in `codecs/sections.py` is camera-only. `H5Cam.sections`, `del_section` and `has()` all iterate it. | New table `GEO_ATTRS` (§4). The camera classes are not touched. |
| R4 | "`MV` = (M, V)" is ambiguous. | Counts are defined in §3. |
| R5 | open3d gives colours as float64 in 0–1. | Store uint8 (D4). The converter rounds `c*255`, which is lossless for ply-sourced u8 colours. Note: Houdini's own `Cd` is float, so this is a documented deviation. |
| R6 | double → f32 is not bit-exact (max error 2.4e-7 m). | SpatialBench compares P with `atol=1e-6`. |
| R7 | "< 0.5 GB" is not reachable losslessly. Projected size is ~0.8 GB. | §5 test plan (D6). |
| R8 | Range-checking ptnum on open would decompress the whole topology. | Checked at write and read time instead (D5, decided). |
| R9 | `H5Base.set_attr` already means root attrs. | Per-level accessors are named `point_attr` / `vertex_attr` / `prim_attr`. |

---

## 1. Houdini model

| level | Houdini meaning | H5Geo group | rows |
|---|---|---|---|
| **detail** | the whole object | root attrs (scalars) + `/detail/*` (tables) + `/meta` (dataset context) | 1 |
| **point** | a position | `/points/<attr>` | `n_points` = M |
| **vertex** | a prim corner referencing one point | `/vertices/<attr>` | `n_vertices` = V |
| **prim** | a polygon made of consecutive vertices | `/prims/<attr>` | `n_prims` = F |

Topology, the prim → vertex → point chain:

- **`/vertices/ptnum`** (V,) int32 holds the point each corner references. The name matches Houdini, where `@ptnum` in a vertex wrangle is exactly this.
- **Fixed-size prims** (all triangles or all quads): set `attrs.prim_size` = 3 or 4. Prim f then owns vertices `[f*prim_size, (f+1)*prim_size)`, and no per-prim array is needed.
- **Mixed polygons:** set `attrs.prim_size = 0` and store **`/prims/vertexcount`** (F,) int32. This is the Houdini intrinsic. Offsets are the cumulative sum, computed on read. Storing counts rather than offsets keeps the per-level length rule uniform and compresses better.
- **Point cloud:** no `ptnum` and no `prim_size`. Then V = F = 0.

Corners are a separate level because hard edges and UV seams give **one point several N or uv values**. open3d's `triangle_uvs` (3F, 2) is exactly vertex-level `uv`.

## 2. Normals, colours and UVs by level

| term | Houdini attr | meaning | open3d source |
|---|---|---|---|
| point normal (ply/open3d "vertex normal") | `points/N` | one smooth normal per point | `vertex_normals` / `pcd.normals` |
| corner normal (Houdini vertex N) | `vertices/N` | per corner; hard edges | — (obj `vn` per corner) |
| face normal | `prims/N` | per polygon | `triangle_normals` |
| point colour | `points/Cd` | u8 RGB | `vertex_colors` / `pcd.colors` |
| corner / face colour | `vertices/Cd`, `prims/Cd` | u8 RGB | — |
| UV | `vertices/uv` (standard), `points/uv` (seam-free) | f32 (·,2) | `triangle_uvs` |

All levels may coexist. Readers use the most specific level present, in the order vertex → point → prim. This is Houdini's precedence.

## 3. File layout (full target)

```
<scene>.geo.h5
  # ===== detail: root attrs =====
  n_points     int64   M (pinned at create)                                  required
  is_metric    bool    default False                                         required
  prim_size    int64   3 / 4 = fixed, 0 = mixed (vertexcount)  present iff topology present
  # ===== detail: /detail =====
  /detail/class_names     str (L,)   name of class id i                       [P2]
  /detail/material_names  str (K,)   material k name (Houdini shop_materialpath equivalent)  [P2]
  /detail/textures/{k}    RESERVED — textures deferred (D11); path name kept free, not implemented
  # ===== points (length M) =====
  /points/P          f32 (M,3)  world metres (same world as H5Cam /T)  required  [P1]
  /points/Cd         u8  (M,3)  RGB                                             [P1]
  /points/N          f32 (M,3)                                                  [P1]
  /points/Alpha      u8  (M,)                                                    [P2]
  /points/uv         f32 (M,2)                                                   [P2]
  /points/pscale     f32 (M,)   point radius (splats / surfels)                  [P2]
  /points/conf       f32 (M,)   confidence 0-1       (h5scene extension)         [P2]
  /points/intensity  f32 (M,)   lidar intensity      (h5scene extension)         [P2]
  /points/class      i32 (M,)   semantic id -> /detail/class_names, -1 = none    [P2]
  /points/id         i32 (M,)   instance id, -1 = none                           [P2]
  # ===== vertices / corners (length V) =====
  /vertices/ptnum    i32 (V,)   referenced point, 0 <= i < M   required iff topology  [P1]
  /vertices/N        f32 (V,3)                                                   [P2]
  /vertices/uv       f32 (V,2)                                                   [P2]
  /vertices/Cd       u8  (V,3)                                                   [P2]
  # ===== prims (length F) =====
  /prims/vertexcount i32 (F,)   only when prim_size == 0; each >= 3              [P2]
  /prims/N           f32 (F,3)  face normal                                      [P2]
  /prims/Cd          u8  (F,3)                                                   [P2]
  /prims/material    i32 (F,)   -> /detail/material_names, -1 = none             [P2]
  /prims/class       i32 (F,)   -> /detail/class_names, -1 = none                [P2]
  /prims/id          i32 (F,)   instance id                                      [P2]
  # ===== groups (Houdini point / prim groups) =====
  /groups/points/<name>  i32 (k,) sorted unique point indices                    [P2]
  /groups/prims/<name>   i32 (k,) sorted unique prim indices                     [P2]
  # ===== custom attributes (D10) =====
  /points|vertices|prims/<custom>  dtype in {f32, f16, i32, u8, bool}, (N[, c])  [P2]
  # ===== /meta (H5Cam keys minus seq / src_frame_idxs) =====
  /meta/dataset, /meta/scene, /meta/suffix, /meta/split, /meta/tags/*
```

**Counts.** Only `n_points` is stored. The others are derived:
- V = `len(ptnum)`.
- F = V / prim_size for fixed-size prims, or `len(vertexcount)` for mixed polygons.

**Length rule.** One rule, checked on every write and at open, for standard and custom attributes alike:
- `/points/*` has axis 0 = M;
- `/vertices/*` has axis 0 = V;
- `/prims/*` has axis 0 = F.

Mixed polygons also need `sum(vertexcount) == V`.

**Registries.**
- `ATTR_KEYS = {"n_points": int, "is_metric": bool, "prim_size": int}`.
- `META_KEYS` is derived from H5Cam's (minus `seq` and `src_frame_idxs`).

**Houdini name notes.** Standard Houdini names are used wherever they exist:
- `P`, `Cd`, `N`, `uv`, `Alpha`, `pscale`, `id`, `class`, `ptnum`, `vertexcount`.

`conf`, `intensity` and the int `material` (index into `/detail/material_names`; Houdini uses a string `shop_materialpath`) are h5scene extensions. They are documented as such in `docs/schema.md`.

**Not stored:**
- bbox and centroid (derived);
- object transform (P is already in world coordinates);
- edges and adjacency;
- other prim types such as polylines or volumes. A `/prims/type` attribute is reserved for these.

## 4. Attribute table (`codecs/sections.py` → `GEO_ATTRS`)

```
GeoSpec = namedtuple("GeoSpec", "level dtype cols compression compression_opts shuffle")
GEO_ATTRS = {                       # level      dtype cols filters
  "points/P":           ("points",   f32, 3, gzip4+shuffle),
  "points/Cd":          ("points",   u8,  3, gzip4),
  "points/N":           ("points",   f32, 3, gzip4+shuffle),
  "points/Alpha":       ("points",   u8,  1, gzip4),
  "points/uv":          ("points",   f32, 2, gzip4+shuffle),
  "points/pscale":      ("points",   f32, 1, gzip4+shuffle),
  "points/conf":        ("points",   f32, 1, gzip4+shuffle),
  "points/intensity":   ("points",   f32, 1, gzip4+shuffle),
  "points/class":       ("points",   i32, 1, gzip4+shuffle),
  "points/id":          ("points",   i32, 1, gzip4+shuffle),
  "vertices/ptnum":     ("vertices", i32, 1, gzip4+shuffle),
  "vertices/N":         ("vertices", f32, 3, gzip4+shuffle),
  "vertices/uv":        ("vertices", f32, 2, gzip4+shuffle),
  "vertices/Cd":        ("vertices", u8,  3, gzip4),
  "prims/vertexcount":  ("prims",    i32, 1, gzip4+shuffle),
  "prims/N":            ("prims",    f32, 3, gzip4+shuffle),
  "prims/Cd":           ("prims",    u8,  3, gzip4),
  "prims/material":     ("prims",    i32, 1, gzip4+shuffle),
  "prims/class":        ("prims",    i32, 1, gzip4+shuffle),
  "prims/id":           ("prims",    i32, 1, gzip4+shuffle),
}
CUSTOM_DTYPES = (f32, f16, i32, u8, bool)   # multi-byte: gzip4+shuffle; 1-byte: gzip4
GEO_CHUNK_ROWS = 65536                      # chunk = (min(N, 65536)[, cols])  — confirmed by §5 sweep
```

- `cols = 1` means the attribute is stored 1-D, (N,).
- **Filter rule:**
  - multi-byte types use gzip4 + shuffle;
  - 1-byte types use gzip4 only, since shuffle has nothing to split (the same reasoning as `mask`).
- **Groups** are int32 with gzip4 + shuffle.
- **Detail string tables** and **texture bytes** are stored raw (textures are already compressed).
- **Standard names fix dtype, cols and meaning.** Writes cast safely:
  - f64 → f32;
  - any int → i32, with a range check.

  Anything else is rejected.
- `filter_args()` / `has()` gain a `table=` parameter. Its default stays `SECTIONS`, so H5Cam and H5CamRig are unchanged.

## 5. Compression test plan (D6)

**Goal:** pin a lossless policy with evidence, and prove the real 43-scene total stays under 1.8 GB (expected ~0.8 GB).

**Execution:** an agent runs this section as an experiment brief, picks the policy by the rule in §5.0, and writes the report. The owner reviews the report before `GEO_ATTRS` is frozen.

### 5.0 Options to test and the decision rule

| id | option | applies to | why test it |
|---|---|---|---|
| C1 | gzip4 + shuffle (multi-byte), gzip4 (1-byte) | all | **baseline**: the current pick and the existing library convention |
| C2 | gzip1 / gzip6 / gzip9, ± shuffle | all | size vs speed curve around the baseline |
| C3 | lzf ± shuffle | all | fast reads; built into h5py with no new dependency |
| C4 | no filter | all | lower bound for read time (the K/T precedent: small arrays may be better raw) |
| C5 | chunk rows 16k / 64k / 256k / 1M | all | filter efficiency and read time for whole-array reads |
| C6 | ptnum flat (V,) vs (F,3) | ptnum | confirm the layout doesn't change the ratio |
| C7 | per-source exception (plain gzip for grid-like P, e.g. 7scenes) | P | quantify whether a per-attribute or per-file choice is worth it |
| C8 | *(info only)* f16 N, quantised P, zstd via `hdf5plugin` | N, P | reported for reference only; **not eligible** (lossy, or a new dependency) |

**Decision rule** (applied per attribute, over all samples):

1. **Lossless.** Every eligible cell must round-trip exactly. Any cell that doesn't is discarded.
2. **Size:** keep the cells within **5 %** of the smallest total stored size.
3. **Read speed:** of those, keep the cells whose full read (`ds[:]`) is ≤ **2×** the C4 read time.
4. **Tie-break:** prefer C1 if it survives (a convention shared with the camera filters). Otherwise take the fastest read.
5. **Per-source exceptions (C7)** are adopted only if they save **> 10 %** of the projected 43-scene total.
6. **Global check:** the projected 43-scene total must be < 1.8 GB. If no cell meets this, stop and report to the owner rather than choosing a lossy option.

The agent's deliverables:
- the filled-in tables in `tests/probes/geo/README.md`;
- the chosen `GEO_ATTRS` rows, with one line of justification each;
- the probe `.npy` files (§5.3).

### 5.1 Evidence so far (5 sample plys, MB, the pick in bold)

| sample | attr | raw | gzip4 | **gzip4+shuffle** | lzf+shuffle |
|---|---|---|---|---|---|
| scannetpp (0.35M pts, 0.71M tris) | P | 4.0 | 3.2 | **2.7** | 3.1 |
| | ptnum (measured as (F,3), same byte order) | 8.1 | 3.3 | **3.0** | 3.6 |
| | Cd | 1.0 | **0.8** | 0.8 | 1.0 |
| dtu (1M) | P | 11.4 | 10.6 | **9.5** | 10.5 |
| | Cd | 2.9 | **2.7** | 2.7 | 2.8 |
| nrgbd (1M) | P | 11.4 | 10.5 | **9.6** | 10.6 |
| | Cd | 2.9 | **2.4** | 2.4 | 2.6 |
| hiroom (1.2M) | P | 13.8 | 12.3 | **10.7** | 12.0 |
| | Cd | 3.4 | **2.2** | 2.2 | 2.7 |
| 7scenes (1M) | P | 11.4 | *6.5* | 7.9 | 9.7 |
| | N | 11.4 | 10.6 | **9.6** | 10.6 |
| | Cd | 2.9 | **2.4** | 2.4 | 2.7 |

- The h5/ply ratio is 0.40–0.47, which projects to ~0.8 GB.
- 7scenes P lies on an f32-exact grid, so plain gzip wins there. That is not worth a per-source policy, but the sweep below re-checks it.

### 5.2 Step 1 — benchmark sweep (one-off, not in the pytest suite)

- **Script:** `tests/smoke_geo_compression.py`. The `smoke_` prefix keeps pytest from collecting it, the same convention as `smoke_image_formats.py`. Pixi task: `pixi run smoke-geo`.
  - It takes a directory of plys as an argument (default: `requirements_gt_geometry/sample_plys`).
  - It has its own ~30-line binary-ply parser **inside the script**, not in the library (D7).
- **Matrix:** for each attribute (P, N, Cd, ptnum):

  | dimension | values |
  |---|---|
  | filter | none / gzip1 / gzip4 / gzip6 / gzip9, each × shuffle on/off; lzf ± shuffle |
  | chunk rows | 16k / 64k / 256k |
  | ptnum layout | flat (V,) vs (F,3) — confirm they are equal |

- **Metrics per cell:**
  - stored bytes;
  - ratio to raw;
  - write time;
  - **full-read time**, i.e. `ds[:]`, which is the consumer's only access pattern;
  - a bit-exact round-trip assert.
- **Output:**
  - a markdown table written to `tests/probes/geo/README.md`, the same role as `tests/probes/README.md` plays for rgb/depth;
  - the per-source h5/ply ratio;
  - a **projected 43-scene total**: each source's ratio × that source's total ply bytes, using the per-source sizes in the requirement table.
- **Policy choice:** apply the §5.0 decision rule. If a cell other than C1 wins, update `GEO_ATTRS` before Phase 1 ships.

### 5.3 Step 2 — real-data probes (pinned, in the pytest suite)

- **Probe set:** a slice of **50k points from each of the 5 sample plys** (scannetpp: the submesh whose faces use only those points, re-indexed), saved as `.npy` under `tests/probes/geo/<source>/`.
  - It totals about 7 MB, and no ply parser is needed at test time.
  - The probes are generated once by the smoke script (`--write-probes`).
- **Tests** in `tests/test_compression_policy.py`, each docstring citing the §5.2 numbers:
  - `test_geo_P_N_ptnum_gzip_shuffle`: asserts the filters on disk.
  - `test_geo_Cd_gzip_no_shuffle`: asserts the filters on disk.
  - `test_geo_probe_size_budget[source]`: stored size ≤ the measured size × 1.05, and < 0.6 × raw. Skipped if the probes are missing.
  - `test_geo_synthetic_size`: a deterministic grid mesh plus a noisy point cloud (seeded) must stay under a fixed byte budget. This runs even without probes.

### 5.4 Step 3 — full-set acceptance (SpatialBench side)

The converter run over all 43 scenes reports:
- the total and per-source h5 bytes, against the ply bytes;
- the conversion time;
- the full-read time.

**Pass** requires all of:
- total < 1.8 GB (expected ≈ 0.8 GB);
- every scene round-trips (P within `atol=1e-6`; Cd / N / ptnum exact);
- mesh sample counts match.

The result goes into `tests/probes/geo/README.md`.

### 5.5 Out of scope unless the owner asks

Lossy options (f16 N, quantised P) and zstd via `hdf5plugin` (a new dependency). The sweep can report them as information-only columns if useful.

## 6. Public API

### Construction

```
H5Geo.create(path, P, *, is_metric=False)  -> H5Geo   # pins n_points = len(P); writes /points/P
H5Geo.open(path, mode="r")                 -> H5Geo
```

Construction follows the H5Cam pattern:
1. `object.__new__` bypasses the open-only `__init__`;
2. `H5Base.__init__(create=True)` creates the file;
3. the level groups are created;
4. `_abort_create()` runs on any failure.

### Topology (written as one unit)

```
geo.set_triangles(tri)               # (F,3) int -> prim_size=3, ptnum = tri.ravel()           [P1]
geo.triangles() -> (F,3) i32         # prim_size 3 (or 0 with all vertexcount == 3); else raise [P1]
geo.set_quads(q)                     # (F,4) int -> prim_size=4                                 [P2]
geo.set_polygons(ptnum, vertexcount) # (V,), (F,) each >= 3 -> prim_size=0                      [P2]
geo.polygons() -> (ptnum (V,), vertexcount (F,))                                                [P2]
geo.del_topology()                   # removes ptnum, vertexcount, prim_size, all /vertices/*, /prims/*, prim groups
```

- **Validation before writing:** every index is in 0..M-1, every count is ≥ 3, `sum(vertexcount) == V`, and V fits in int32.
- **Changing topology** while vertex or prim attributes exist raises an error. Call `del_topology()` first, the same pattern as deleting `depth_conf` before `depth`.

### Attributes

Generic accessors, keyed by Houdini name:

```
geo.point_attr("Cd") / geo.set_point_attr("Cd", a) / geo.del_point_attr("Cd")
geo.vertex_attr("uv") / set_vertex_attr / del_vertex_attr
geo.prim_attr("N")    / set_prim_attr   / del_prim_attr
geo.point_attrs / vertex_attrs / prim_attrs   -> set of present names
```

**Shortcuts:**
- Phase 1: `geo.P()` / `geo.set_P(p)`, `geo.Cd()` / `geo.set_Cd(c)`, `geo.N()` / `geo.set_N(n)` (point level).
- Phase 2 reaches the other levels through the generic accessors.

**Write rules.** All are checked before the old dataset is deleted, so a rejected write keeps the old data:

- **rows:** the row count equals the level count;
- **casting:** f64 → f32 and int → i32 (range-checked) are allowed. Float → int and bool → float are rejected;
- **Cd / Alpha:** must be **uint8**, and float is rejected (D4). The converter does `np.round(c*255).astype(np.uint8)`;
- **cols:** must match the table;
- **floats:** no NaN or inf;
- **ids** (`class` / `id` / `material`): must be ≥ -1, and below the length of the name table when one exists;
- **P:** a replacement must keep M;
- **vertex / prim attributes:** need topology to exist.

**Derived helpers** (never stored):
- `bbox()`;
- the Phase 2 normal helpers: point N from faces, and face N from P.

There is no surface sampling in the library (D12).

### Detail tables, groups, materials [P2]

```
geo.set_class_names([...]) / class_names()
geo.set_material_names([...]) / material_names()      # names only; textures deferred (D11)
geo.set_point_group(name, idx_or_mask) / point_group(name) / del_point_group(name)
geo.set_prim_group(name, idx_or_mask)  / prim_group(name)  / del_prim_group(name)
```

### Properties

- `n_points`, `n_vertices`, `n_prims`, `prim_size`
- `has_topology`, `is_triangles`
- `is_metric`, `has(name)`
- `dataset` / `scene` / `suffix` / `split` / `tags`
- `__len__` (= M) and `__repr__` (shows each level and its attributes)

### Validation

**At open** (O(1) in data size). Open raises `ValueError` and closes the handle when any of these fails:

1. The root attributes are present. `prim_size` must be present exactly when `ptnum` is.
2. `/points/P` exists with shape (M, 3) f32.
3. `prim_size` is in {0, 3, 4}. For prim_size 3 or 4, V % prim_size == 0. `vertexcount` must be present exactly when prim_size == 0.
4. Every dataset has a known (or allowed custom) name and the right dtype and cols, and the length rule holds.

**At write and read time** (decided, D5). These value-range checks run when the array is already in memory:
- `ptnum` range;
- `vertexcount` (each ≥ 3, sum = V);
- ids;
- group indices.

## 7. open3d → H5Geo (in SpatialBench, not this library — D7)

| open3d | H5Geo |
|---|---|
| `mesh.vertices` / `pcd.points` | `H5Geo.create(path, P)` |
| `mesh.vertex_colors` / `pcd.colors` | `set_Cd(np.round(c*255).astype(np.uint8))` |
| `mesh.vertex_normals` / `pcd.normals` | `set_N(n)` |
| `mesh.triangles` | `set_triangles(t)` |
| `mesh.triangle_normals` | `set_prim_attr("N", …)` [P2] |
| `mesh.triangle_uvs` (3F,2) | `set_vertex_attr("uv", …)` (already in corner order) [P2] |
| `triangle_material_ids` + `textures` | `set_prim_attr("material", …)` + `set_materials(…)` [P2] |

Going back to open3d:
```
TriangleMesh(Vector3dVector(P.astype(f64)), Vector3iVector(geo.triangles()))
```
Colours come back via `Cd / 255.0`.

## 8. Tests (`tests/test_geo.py`)

**Phase 1:**
1. **Triangle mesh round-trip** (P / Cd / N / triangles): create → close → open returns `array_equal` data, correct counts, and `prim_size == 3`.
2. **Point-cloud round-trip:** V = F = 0, and `triangles()` raises `KeyError`.
3. **f64 P:** the stored value equals `astype(f32)`. **Float Cd is rejected.**
4. **Bad indices:**
   - `set_triangles` rejects an index of M, an index of -1, and a float array;
   - a bad `ptnum` written through raw h5py raises on `triangles()`, and open still succeeds (D5).
5. **Missing P or a missing root attribute:** open raises and the handle is closed.
6. **Length rule:** Cd with M ± 1 rows is rejected on write. A hand-edited Cd is rejected at open.
7. **Unknown datasets** under `/points` are rejected at open (unless D10 allows custom attributes).
8. **Failed operations:**
   - a failed write keeps the old data;
   - a failed create removes the file.
9. **Meta:**
   - the derived keys work;
   - `seq` / `src_frame_idxs` are rejected;
   - `is_metric` defaults to False.
10. **Compression tests** from §5.3.

**Phase 2:**
11. **Quads and mixed polygons:**
    - mixed `vertexcount` round-trips;
    - `sum != V` and any count < 3 are rejected;
    - `triangles()` raises on mixed sizes.
12. **Per-level attributes:** vertex-level N / uv / Cd have length V, and prim-level N / Cd / class / material have length F.
13. **Topology change:** changing topology while attributes exist raises; it works after `del_topology()`.
14. **Ids:** `class` / `material` must be below their name tables, and -1 is allowed.
15. **Groups:** masks and indices both come back as sorted indices, and out-of-range indices are rejected.
16. **Textures:** deferred (D11); no test.
17. **Custom attributes:** the dtype whitelist and length rule are enforced, and a name that clashes with a standard one is rejected.

## 9. Docs to sync (CLAUDE.md requirement)

- **`h5scene/__init__.py` docstring:**
  - H5Geo;
  - the level table;
  - layout and usage.
- **`docs/schema.md`** (Chinese), a new `## H5Geo` section:
  - levels and topology;
  - the attribute table, with Houdini names and h5scene extensions marked;
  - the length rule;
  - the checks;
  - the compression table.
- **`README.md`:** usage examples for a point cloud and a mesh.
- **`CLAUDE.md`:**
  - an H5Geo bullet and `GEO_ATTRS`;
  - the invariants: vertex = corner; length rule per level; topology written as a unit; no prims means a point cloud; Cd stored as u8 (a deviation from Houdini).
- **`pyproject.toml`:** the `smoke-geo` pixi task.

## 10. Work order

**Phase 1:**
1. **Compression sweep (§5.2).** This settles the gzip level and chunk rows, writes the probes, and writes `tests/probes/geo/README.md`.
2. **`sections.py`:**
   - `GeoSpec`;
   - the full `GEO_ATTRS` table;
   - the `table=` parameter.

   The existing suite must still pass unchanged.
3. **`h5geo.py`:**
   - create / open and validation;
   - the generic accessors;
   - `set_triangles` / `triangles` / `del_topology`;
   - the P / Cd / N shortcuts.
4. **Tests and docs:** the Phase 1 tests, the §5.3 compression tests, and the docs.
5. **Hand-off to SpatialBench:** the converter, plus the 43-scene acceptance run (§5.4).

**Phase 2:**

6. Quads and polygons.
7. Detail tables (class and material names; textures deferred).
8. Groups.
9. Custom attributes.
10. Normal helpers.
11. Phase 2 tests and docs.

## 11. Library dependencies

There are no new runtime dependencies (numpy, h5py, opencv-python as now). The PLY parser lives only in the smoke script.

## 12. Decisions

| id | question | status |
|---|---|---|
| D1 | Class name | **Decided: `H5Geo`** |
| D8 | Model | **Decided: Houdini levels** (vertex = corner) |
| D9 | Attribute names | **Decided: Houdini** (P, Cd, N, uv, Alpha, pscale, id, class, ptnum, vertexcount) |
| D13 | Scope | **Decided: full structure**, delivered in two phases on one fixed schema |
| D3 | `is_metric` | **Decided: always written, default False** |
| D4 | Colours | **Decided: uint8 only** (the converter rounds `c*255`) |
| D5 | Value-range checks | **Decided: at write and read time;** open checks shapes and dtypes only |
| D7 | PLY reader | **Decided: not in this library** (the converter uses open3d; the smoke script has a private parser) |
| D6 | Compression | **Decided: a delegated experiment.** An agent runs the §5 brief and picks a policy by the §5.0 decision rule. The owner reviews the resulting report. |
| D2 | Stored counts | **Decided:** `n_points` + `prim_size` only; V and F are derived |
| D2b | create | **Decided:** `create(path, P)` |
| D10 | Custom attributes | **Decided: allowed**, with a dtype whitelist (f32/f16/i32/u8/bool) and the per-level length rule |
| D11 | Textures | **Decided: deferred.** This request does not need them (no GT ply has textures). `material_names` / `prims/material` stay in Phase 2. `/detail/textures/` is reserved and will be decided in a later request. |
| D12 | Surface sampling in the library | **Decided: no.** SpatialBench samples with open3d. |
| — | Polylines / curves / volumes; linking geo into the rig | Out of scope (a `/prims/type` attribute is reserved) |
