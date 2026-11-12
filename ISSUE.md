# ISSUE.md — h5scene review

✅ fixed (with a regression test or a diff) · ⏭️ dropped on purpose · ❌ false positive · 🔲 open
**[verified]** = reproduced with a scratch script.

## Closed: first review #1–#35 (2026-11-12)

| # | | Resolution |
|---|---|---|
| 1 | ✅ | Rig VDS per-frame blocks now select camera frame `t` (`VirtualSource(path, name, (T, h, w))[t]`), not frame 0. |
| 2 | ✅ | Rig build rejects downsampled depth/depth_conf, since a VDS block can't upscale. |
| 3 | ✅ | `set_rgb` checks that the frame count equals T. |
| 4 | ✅ | `set_meta` validates before it deletes the old value. |
| 5 | ✅ | Rig build validates first, writes a temp file, then `os.replace`s it, so a failed build leaves the existing output alone (but see #36). |
| 6 | ✅ | rgb must be in all cameras or none, and it counts in the section-set check. |
| 7 | ✅ | Rig `is_metric` goes through `codecs.sections.is_metric` (attrs first, legacy fallback, default False). |
| 8 | ✅ | The public API is RGB end to end; cv2's BGR is converted only at the `codecs.rgb` boundary. |
| 9 | ✅ | Every integer depth dtype is treated as mm. |
| 10 | ✅ | The documented broadcasts are implemented: mask `(H,W)`/`(T,)`, rgb `(H,W,3)` (T=1), depth/conf `(H,W)` (T=1). |
| 11 | ✅ | `create/open/build` classmethods exist, `open` takes `"r"`/`"r+"`, and the README is rewritten. |
| 12 | ✅ | The rig inherits `split`/`tags` only when every camera agrees. |
| 13 | ✅ | `src_frame_idxs` (was `frame_indices`) must have length T and isn't stored when it equals `arange(T)`. |
| 14 | ✅ | The docstring now says DQT detection only recognises cv2's 8 qualities; anything else is re-encoded at q95. |
| 15 | ✅ | numpy scalars are accepted for `bool`/`int`/`float` keys (`_NUM_EQUIV`). |
| 16 | ✅ | The stale docstrings were rewritten during the module split. |
| 17 | ✅ | The dead `_io.py`/`rgb.py`/`convert.py` were removed; `codecs/` is the single code path. |
| 18 | ✅ | Empty or duplicate `cam_paths` raise a clear error. |
| 19 | ✅ | The rig's rgb size check reads the JPEG SOF header (`jpeg_hw`) instead of decoding. |
| 20 | ✅ | The rig stores a per-camera fingerprint in `attrs.cam_fp` and checks it on open (but see #39, #40). |
| 21 | ✅ | `H5Cam.open` checks section shapes against THW. |
| 22 | ✅ | `has()` tolerates files with no `/rgb` or `/meta`. |
| 23 | ✅ | If a subclass constructor fails after create, `_abort_create()` closes the handle and removes the half-made file. |
| 24 | ✅ | `tags` values must be `str`; they are no longer silently `str()`-ed. |
| 25 | ✅ | `tests/test_issue_bugs.py` pins the bug fixes, and `sys.path.insert` is gone. |
| 26 | ✅ | `c0/c1.h5` were removed, `*.h5` is gitignored, and pillow/scikit-image live only in the `imgtest` feature. |
| 27 | ✅ | The modules are split by role: `codecs/{rgb,depth,pose,sections}` + `h5base/h5cam/h5camrig`. |
| 28 | ✅ | One declarative `SECTIONS` table drives the writers, the VDS build, `has()` and the compression test. |
| 29 | ⏭️ | The reader mixin was dropped, together with #33. |
| 30 | ✅ | The classmethods are the recommended entry points; `__init__` stays for compatibility. |
| 31 | ✅ | `set_meta` is defined once in the base and takes a single-key and a multi-key form; subclasses only hook `_meta_value` (but see #37, #38). |
| 32 | ✅ | `MetaSpec(type, dtype, length="T")` is checked in the base, and the rig's `META_KEYS` is derived from `H5Cam`'s. |
| 33 | ⏭️ | The in-memory `CamRig` was built and then reverted at the user's request; the VDS file is the only rig form. |
| 34 | ✅ | `__len__`, `sections` and `__repr__` are done on both classes; batch reads (`depth(slice)`) were not done. |
| 35 | ✅ | The tests mirror the modules and `--ignore` is gone; smoke scripts stay in `tests/` (user rule), and pytest doesn't collect them because of the `smoke_*.py` name. |

## Closed: independent audit (18 items, 2026-11-12)

- ✅ **8 fixed:** rig build handle leak · `H5Cam.open` caught only `ValueError` · unguarded close in `_abort_create`/`__exit__` · exactly 65.535 m now written as 0 · empty rgb input · unchecked `imencode` · 3 dead-code spots · `requires-python >=3.11`.
- ❌ **3 false positives:** `Image.NEAREST` still exists in pillow 13 (×2), and `filter_args` is used.
- ⏭️ **Kept on purpose:** the non-atomic `set_rgb` replace (noted in the code), and the unused `T` parameter of `normalize_rgb`/`to_mm`, kept for signature symmetry.

## Open: found in review (2026-11-13, at `cca9d3f`)

### Bugs

36. ✅ A successful build whose output path is one of its own cameras is rejected in step 0 of `_build` (an `os.replace` would have overwritten the camera). `test_rig_output_path_must_not_be_a_camera`.
37. ✅ The multi-key `set_meta` validates EVERY key before the first write; a bad second key leaves nothing written. `test_set_meta_multi_key_validates_all_before_writing`.
38. ✅ Mixing the two `set_meta` forms (positional name AND keywords) raises instead of silently dropping the keywords. `test_set_meta_mixing_forms_is_an_error`.
39. ✅ The fingerprint's (T, H, W) is now read from the CAMERA file's `attrs.THW` at open and compared (plus size); the stored `cam_fp` is `(T, H, W, size)` and the error names what actually changed. Legacy 5-column rows (with mtime) still verify.
40. ✅ `cam_files` / the VDS sources / the ExternalLinks are stored RELATIVE to the rig file's directory (h5py resolves both the same way — verified empirically from two different CWDs), so a rig copied or moved together with its cameras opens fine (byte-identical content, new mtime); the fingerprint is content-based (THW + size), not mtime. `test_rig_survives_cp_and_moves_with_cameras`. A camera moved away or rewritten still raises `FileNotFoundError`/`ValueError`.
41. ✅ `set_depth_conf_01` now takes 0-1 floats ONLY (integers are rejected — they are never a 0-1 confidence, no more dtype-max scaling). `test_conf_rejects_integers_and_out_of_range` (supersedes `test_conf_uint8_scaled_by_dtype_max`).
42. ✅ `H5CamRig.build` creates the output directory (like `H5Cam.create`). `test_rig_build_creates_output_dir`.

### Robustness / API

43. ✅ **Depth resolution can't be changed in place.** `del_section(name)` added (rgb: all frames; the rest: the whole dataset); depth cannot go while depth_conf shares its (h, w) — delete depth_conf first, then set_depth at the new (h, w) works. (test_del_section) While `depth_conf` exists at the old `(h, w)`, `set_depth` at a new `(h, w)` is rejected, and the same holds the other way round. There is also no way to delete a section. Add `del_section(name)`, or a way to set both together.
44. ✅ **`jpeg_hw` misses some cases.** SOF list now covers C0–CF; a truncated SOF (or a malformed ln<4) returns None instead of IndexError (test_jpeg_hw_survives_truncated_and_odd_headers scans every truncation point).
45. ✅ **`normalize_rgb` fully decodes every input frame.** A q<=95 jpeg is size-checked from the SOF header only and kept verbatim; only re-encoded frames are decoded (test_normalize_kept_frames_are_not_decoded). Use `jpeg_hw` for the size check, and decode only the frames that get re-encoded.
46. ✅ **Rig camera handles are opened outside the `try`.** Cameras are opened inside the try that closes them in a `finally` — a k-th open failure closes cameras 0..k-1 deterministically. `cams = [H5Cam(p) for p in absp]` sits before the `try`, so if the k-th open fails, cameras 0..k-1 stay open until GC closes them. That isn't a leak in practice, but the close isn't deterministic.

### Simplification / stale bits

47. ✅ **The rig build duplicates base code.** The temp file is written through a base-class handle (`H5Base.__init__(…, create=True)` on the temp path; `_abort_create` on failure) — attrs go through `set_attr`, /meta through the base `set_meta` (validated before any write); the hand-rolled /meta serialization and its dead ndarray branch are gone.
48. ✅ **The registry is half `MetaSpec`.** Every META_KEYS entry is a MetaSpec (H5Cam + the rig's suffix); the tags check (str keys AND values) moved into MetaSpec.check; the legacy plain-type branch of `_check_meta_spec` is gone (test_meta_registry_is_all_metasp). `H5Cam.META_KEYS` mixes plain `str` types with `MetaSpec`, while the docs say the values are `MetaSpec`, so `_check_meta_spec` keeps a "legacy" plain-type branch for that. The `tags` value check lives in `H5Base` instead of `MetaSpec`, and `tags` keys aren't checked at all. Make every entry a `MetaSpec` and move the dict check into `MetaSpec.check`.
49. ✅ **Dead code and stale text** — all fixed in the two review commits (`9967b49` codecs/h5cam/docs + this commit: rig `h_`).
    - **Unused:**
      - `h_` in `H5CamRig._validate`.
      - `RESIZABLE`, exported but unused, with a stale "read decoder name" comment.
      - `rgb_frame_hw`.
      - The `has(h, "is_metric")` branch (not a section).
    - **No-ops and duplicates:**
      - The `H5Cam.src_frame_idxs` body `None if fi is None else fi`.
      - The duplicated `# ---- writers` comment in `h5cam.py`.
    - **Stale or wrong text:**
      - The `sections.py` docstring says the K/T broadcast is "in each class's set_*"; it lives in `codecs/pose`.
      - `broadcast_matrix` says "pose must be …" for K too.
      - The `__init__.py` layout docstring leaves out `attrs.cam_fp`.
      - The README usage still shows the constructor form `H5Cam("cam.h5", T, H, W)` instead of `H5Cam.create`, and has a stray `from h5scene import H5Base` inside the `with` block.

## Changes: naming guards + /img rename (2026-11-14)

API-breaking changes (old `.h5` files with the `/rgb` layout no longer open); unit/range
mistakes now raise instead of being guessed:

1. **`set_depth` → `set_depth_mm` (integer MM only) + `set_depth_meters` (float METERS only).**
   The unit is fixed by the entry point, never inferred from the dtype: a float into
   `set_depth_mm` and an integer into `set_depth_meters` raise `TypeError` pointing at
   the other writer. Reads: `depth(t, meters=…)` → `depth_mm(t)` / `depth_meters(t)` (cam
   and rig).
2. **`set_depth_conf` → `set_depth_conf_01` (0-1 float only).** Integers raise `TypeError`
   (closes #41 — no more silent dtype-max scaling), out-of-range floats raise `ValueError`.
3. **`set_pose` → `set_pose_c2w` / `pose(t)` → `pose_c2w(t)`** (cam and rig). The STORED
   convention is unchanged: camera-to-world (p_world = c2w @ p_cam needs one multiply,
   and the rig's cross-camera transforms are `c2w_j @ inv(c2w_i)`).
4. **`/rgb/{t}` → `/img/{t}`** (cam and rig, incl. the rig's ExternalLink targets and
   `has("img")` / `sections` / `del_section("img")`); docs + tests + `probes/rig_scan.py`
   updated. BREAKING for existing files.
5. **BGR access for cv2 users:** `bgr(t)` / `bgr(t, n)` = `rgb(...)[:, :, ::-1].copy()`
   (read side); `set_bgr` = uint8 BGR ARRAYS ONLY (no image bytes / paths — the colour
   order on disk is unknowable), channels reversed then stored through the `set_rgb`
   policy. Storage is still RGB-semantics jpg.
6. **`codecs/depth.py`:** `to_mm` (integers only) / `to_meters_mm` (floats only)
   replace the dtype-sniffing `to_mm`; `to_conf` (0-1 floats only).

Tests: `test_set_bgr_and_unit_guards` (the guards + the red/green swap round trip),
`test_conf_rejects_integers_and_out_of_range`, the existing round-trip tests updated to
the new names and `/img`.
