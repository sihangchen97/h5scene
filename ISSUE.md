# ISSUE.md — h5scene review

✅ fixed · ⏭️ dropped on purpose · ❌ false positive · 🔲 open
**[verified]** = reproduced with a scratch script.

## Closed: first review #1–#35 (2026-11-12)

| # | | Resolution |
|---|---|---|
| 1 | ✅ | Rig VDS blocks select camera frame `t`, not frame 0. |
| 2 | ✅ | Rig build rejects downsampled depth/depth_conf. |
| 3 | ✅ | `set_rgb` requires frame count == T. |
| 4 | ✅ | `set_meta` validates before deleting the old value. |
| 5 | ✅ | Rig build is validate → temp file → `os.replace`. |
| 6 | ✅ | rgb must be in all cameras or none. |
| 7 | ✅ | Rig `is_metric` goes through `codecs.sections.is_metric`. |
| 8 | ✅ | Public API is RGB; BGR exists only inside `codecs.rgb`. |
| 9 | ✅ | Integer depth dtypes are mm (later superseded by the explicit-unit writers). |
| 10 | ✅ | The documented mask/rgb/depth broadcasts are implemented. |
| 11 | ✅ | `create/open/build` classmethods exist and the README matches. |
| 12 | ✅ | Rig inherits `split`/`tags` only when all cameras agree. |
| 13 | ✅ | `src_frame_idxs` must be length T and is dropped when it equals `arange(T)`. |
| 14 | ✅ | Docstring states DQT detection only knows cv2's 8 qualities. |
| 15 | ✅ | numpy scalars are accepted for `bool`/`int`/`float` keys. |
| 16 | ✅ | Stale docstrings rewritten during the module split. |
| 17 | ✅ | Dead `_io.py`/`rgb.py`/`convert.py` removed. |
| 18 | ✅ | Empty or duplicate `cam_paths` raise. |
| 19 | ✅ | Rig rgb size check reads the SOF header instead of decoding. |
| 20 | ✅ | Rig stores and verifies a per-camera fingerprint (`attrs.cam_fp`). |
| 21 | ✅ | `H5Cam.open` checks section shapes against THW. |
| 22 | ✅ | `has()` tolerates files without `/img` or `/meta`. |
| 23 | ✅ | `_abort_create()` closes and removes a half-made file. |
| 24 | ✅ | `tags` values must be `str`. |
| 25 | ✅ | `tests/test_issue_bugs.py` pins the fixes. |
| 26 | ✅ | Stray `.h5` removed and gitignored; image-test deps isolated in `imgtest`. |
| 27 | ✅ | Modules split by role (`codecs/*`, `h5base/h5cam/h5camrig`). |
| 28 | ✅ | One `SECTIONS` table drives writers, VDS build, `has()` and the compression test. |
| 29 | ⏭️ | Reader mixin dropped together with #33. |
| 30 | ✅ | Classmethods are the entry points; `__init__` kept for compatibility. |
| 31 | ✅ | `set_meta` is defined once in the base; subclasses hook `_meta_value`. |
| 32 | ✅ | `MetaSpec` checked in the base; rig `META_KEYS` derived from `H5Cam`'s. |
| 33 | ⏭️ | In-memory `CamRig` reverted at the user's request; the VDS file is the only rig form. |
| 34 | ✅ | `__len__`/`sections`/`__repr__` added (batch reads not done). |
| 35 | ✅ | Tests mirror the modules; smoke scripts named `smoke_*.py` so pytest skips them. |

## Closed: independent audit (2026-11-12)

- ✅ 8 fixed (handle leak, open exceptions, guarded close, 65.535 m → 0, empty rgb, unchecked `imencode`, dead code, `requires-python >=3.11`).
- ❌ 3 false positives (`Image.NEAREST` ×2, `filter_args` is used).
- ⏭️ Kept on purpose: the non-atomic `set_rgb` replace and the unused `T` parameter of `normalize_rgb`/`to_mm`.

## Closed: second review #36–#49 (2026-11-13)

| # | | Resolution |
|---|---|---|
| 36 | ✅ | Rig build rejects an output path that is one of its cameras. |
| 37 | ✅ | Multi-key `set_meta` validates every key before the first write. |
| 38 | ✅ | Mixing the positional and keyword `set_meta` forms raises. |
| 39 | ✅ | Fingerprint is `(T, H, W, size)` with THW read from the camera at open (legacy 5-col rows still verified). |
| 40 | ✅ | Camera refs are relative to the rig dir and the fingerprint is content-based, so a rig survives `cp`/move with its cameras. |
| 41 | ✅ | `set_depth_conf_01` accepts 0-1 floats only. |
| 42 | ✅ | Rig build creates the output dir (only once validation passes). |
| 43 | ✅ | `del_section(name)` added (but see #51). |
| 44 | ✅ | `jpeg_hw` survives truncated/malformed headers (but see #50). |
| 45 | ✅ | `normalize_rgb` decodes only the frames it re-encodes. |
| 46 | ✅ | Rig build closes already-opened cameras deterministically on failure. |
| 47 | ✅ | Rig temp file is written through an `H5Base` handle (`set_attr`/`set_meta`). |
| 48 | ✅ | Every `META_KEYS` entry is a `MetaSpec`; the dict check lives in `MetaSpec.check`. |
| 49 | ✅ | Listed dead code and stale text removed (but see #55). |

## Closed: API changes + review pass 3 (`c42302b`, `e6e64dd`)

- ✅ The unit/range now lives in the name (`set_depth_mm`/`set_depth_meters`, `set_depth_conf_01`, `set_pose_c2w`), `/rgb` was renamed to `/img`, and `bgr`/`set_bgr` were added. **This breaks existing files.**
- ✅ Pass 3 removed a dead `cv2` import, stopped a failed build leaving an empty dir, removed a self-referencing pixi task and deduplicated the camera-path join.

## Open: review pass 4 (2026-11-13, at `e6e64dd`)

### Bugs

50. ✅ `jpeg_hw` treats non-SOF markers as SOF.** C4/C8/CC removed from the SOF set Its marker list is C0–CF, which includes `FFC4` (DHT), `FFC8` (JPG) and `FFCC` (DAC). In a valid 48×64 cv2 JPEG with DHT moved before SOF (allowed by the spec, and some encoders do this), `jpeg_hw` returns `(1, 1281)`. `normalize_rgb` keeps that frame verbatim with this wrong size, so `set_rgb` rejects a good image and the rig size check fails. Fix: exclude C4/C8/CC from the SOF set.
51. ✅ Rig fingerprint misses section edits to a camera.** `_verify_sources` now also checks each mapped section's presence + shape and the /img frames in every camera (test_rig_open_catches_camera_section_edits) `del_section("depth")` on a camera after the rig is built leaves the file size (14336 → 14336, since HDF5 doesn't reclaim the space) and THW unchanged. `H5CamRig.open` then succeeds, `has("depth")` is True, and `depth_mm(0, 1)` silently reads all 0 (the VDS fillvalue). That silent read is exactly what #20/#39 were meant to prevent. Re-storing depth at a smaller `(h, w)` also keeps the size the same (6186 → 6186). Fix: `_verify_sources` already opens each camera to read THW, so it can also check that every rig VDS section exists in the camera with the shape that was mapped (`(T, H, W)` / `FRAME_SHAPE`), and that `/img/{t}` exists when the rig has `/img`.
52. ✅ `del_section("depth")` raised `KeyError` when only `depth_conf` exists.** the guard now checks `"depth" in h` first (the always-true (h, w) comparison is dropped; test_del_section_depth_without_depth_only_conf) `set_depth_conf_01` is allowed without depth, but the guard reads `h["depth"].shape` before it checks `"depth" in h`. The shape comparison is also always true, because `_check_depth_hw` enforces the shared `(h, w)`. Fix: `if name == "depth" and "depth" in h and "depth_conf" in h: raise`.
53. ✅ `pixi run test` failed in the default env** (no pytest there). the task now lives in `[tool.pixi.feature.test.tasks]` (`No module named pytest`). Pass 3 changed the task to `python -m pytest …`, but the task is declared under the default env, which has no pytest. CLAUDE.md still documents `pixi run test`. Fix: move the task to `[tool.pixi.feature.test.tasks]`, which makes `pixi run test` resolve to the test env.

### Simplification / efficiency

54. ✅ **Per-frame VDS blocks replaced by one mapping per camera** (`lay[:, n, :bh, :bw] = VirtualSource(rel, name, (T, bh, bw))`); the fixed/full-size/padded branches in `_write` collapse to one line, pad -> fillvalue (test_rig_vds_one_mapping_per_camera). One mapping, `lay[:, n, :hh, :ww] = VirtualSource(rel, name, shape=(T, hh, ww))`, reads back correctly for a padded camera (the pad region gets the fillvalue). #1 was a wrong `[0]` selection, not a reason for per-frame blocks. The fixed / full-size / padded branches in `_write` collapse into one line (with `shp = FRAME_SHAPE[name]` or `(hh, ww)`), and a padded camera goes from T mappings to 1, which matters because VDS open/read cost grows with the mapping count.
55. ✅ **Stale text and dead bits — all resolved:**
    - CLAUDE.md still describes `/rgb/{t}`, dtype-inferred units ("float = meters, all integer dtypes = mm"), "depth_conf scaling", and per-frame `[t]` blocks.
    - `sections.py`:
      - the "read decoder name per section" comment is still there (#49 said it was removed);
      - the docstring still says "conf scaling" and "no /rgb";
      - `Spec.resizable` is never read.
    - `h5base.py`: the docstrings say the rig is "cam keys minus seq", but `src_frame_idxs` is excluded too.
    - `H5CamRig.META_KEYS` re-adds `suffix`, which `H5Cam` already has, so it is a no-op.
    - `_check_attr_value`: the `isinstance(t, type)` check is always true.
    - `_STR` is duplicated in `h5camrig.py`.
    - `to_conf`'s message reads "got int8 integers are not accepted" (a dash is missing).
    - `set_depth_mm`/`set_depth_meters` repeat the same 4 lines with only the converter different.
56. ✅ **Rig `_validate` reads only a 65536-byte prefix** (`ds[:65536]`, covers a maximal APP1 before SOF) with a full-read fallback when `jpeg_hw` returns None. It loads T·N full frames only to call `jpeg_hw`. Reading a prefix (`ds[:65536]`, which covers a maximal APP1 before SOF) is enough, with a fallback to the full read when that returns None.

## Open: review of the #50–#56 fixes (2026-11-13, working tree on `e6e64dd`)

57. ✅ Regression from #51: no rig with K or T opens.** the section check now compares against `(T, *FRAME_SHAPE[name])` for K/T (test_rig_with_K_and_pose_opens) `_verify_sources` sets `want = FRAME_SHAPE[name]`, which is `(3, 3)`, but the camera stores `(T, 3, 3)`. Repro: two cameras with `set_K(np.eye(3))`, then `H5CamRig.build` + `open` raises "/K is now (3, 3, 3) but the rig mapped (3, 3)". Fix: `want = (T, *FRAME_SHAPE[name])`. The suite stays green because no test opens a rig with K/T.
58. ✅ **All five regression tests now present** (the #54 one added with it):
    - `jpeg_hw` with DHT before SOF;
    - a camera `del_section`/re-store after the build makes `open` raise;
    - a rig with K/T opens (this would have caught #57);
    - `del_section("depth")` with only `depth_conf` stored;
    - the single-block VDS for a padded camera.
59. ✅ **#55 stale text — all sub-items resolved** (the four open bullets at the time are now clean):
    - CLAUDE.md is untouched: it still says `/rgb`, dtype-inferred units, "depth_conf scaling", per-frame `[t]` blocks, `resizable`, and "+ the rig's own `suffix`".
    - `sections.py` still says "resizability" (L3) and "no /rgb" (L49).
    - The `h5camrig.py` L65 comment still says "plus the rig's own suffix".
    - The `h5base.py` docstring has an unbalanced paren: `"src_frame_idxs").)`.
    - The `isinstance(t, type)` check in `_check_attr_value` is still there.
60. ✅ (mooted by #61) Legacy 5-col fingerprint check dropped. The new code compares the stored row only to the camera's THW, not to the rig's own `(T, cam_hw)` as the 4-col branch does. A rig whose `TNHW`/`cam_hw` disagrees with a legacy row is no longer caught.

## Clean-up before release (no legacy support — there are no released files to read)

61. ✅ Drop the legacy 5-column `cam_fp`** — the legacy branch is gone, `cam_fp`/`cam_hw` are required rig attrs (open raises otherwise), and the doc mentions (`__init__.py`, `schema.md`, `h5camrig.py`) are updated (the `(T, H, W, size, mtime_ns)` branch in `_verify_sources`). This also makes #60 moot. Make `cam_fp` required: remove `if "cam_fp" in a`, and add `cam_fp`/`cam_hw` to the "not an H5CamRig file" check. Remove the mentions in `__init__.py` L55-56, `h5camrig.py` L29, docs/schema.md L66 and CLAUDE.md L38.
62. ✅ Drop the legacy `/meta/is_metric` fallback** — `is_metric` reads `attrs["is_metric"]` only (always written by create/build) in `codecs.sections.is_metric`. Read `attrs["is_metric"]` only; it is always written by `create`/`build`, so it can be required at open. Update CLAUDE.md L27.
63. ✅ Drop the meta-key branch in `has()`** — it answers sections + img only (`h.get("img")` may be absent on an image-less file); `meta(name) is not None` covers meta keys (CLAUDE.md updated) `/meta` and `/img` always exist in our files, so `has()` can be `name in h` for sections and `len(h["img"]) > 0` for img. Decide whether `has()` should answer meta keys at all; `meta(name) is not None` already covers that. Update the docstring (L49) and CLAUDE.md L27.
64. ✅ Drop the "legacy absolute paths" comment** — `cam_files` is always relative (`h5camrig.py` L102). `cam_files` is always relative.
65. ✅ **The classmethods are the only constructors** — `__init__` is private-ish (one path each, documented as `create`/`open`-driven), the `H5Cam(p, "r+")` str-in-`t` guard is gone, and the constructor-form test call sites all use `create`/`build`. #30 kept `H5Cam(path, t, h, w)` / `H5Cam(path, mode=…)` / `H5CamRig(out, cam_paths)` "for compatibility". Before release, make `__init__` private-ish (one path each, called by `create`/`open`/`build`) and delete the `H5Cam(p, "r+")` str-in-`t` guard. Update the tests that use the constructor form (e.g. `test_issue_bugs.py` L151).
66. ✅ **Unused `T` parameters removed** from `normalize_rgb`/`to_mm`/`to_meters_mm`/`to_conf` (they were "kept for signature symmetry"; nothing reads them).
67. ✅ **Stale history rewritten** — the `test_issue_bugs.py` header ("line refs in ISSUE.md refer to the old layout") and the size/mtime wording in L144/L149/L151/L323-337; mtime is no longer part of the fingerprint. Keep the behaviour, drop the narrative.
68. ✅ **`rig_scan.py` no longer detects the legacy `attrs.H/W + /meta/images` layout** — the probe requires `attrs.TNHW` (non-rig files become an error row); the legacy branch and its `legacy` count are gone.
69. ✅ **`_tobedelete/` deleted** (it was gitignored/local only) and removed from `rig_scan.py`'s `SKIP`.
70. 🔲 **Collapse ISSUE.md history** once #57–#69 are closed. Keep a short "pre-release review" summary (or move it to a CHANGELOG) instead of the per-pass tables. The API-breaking notes (`/rgb` → `/img`, renamed setters) no longer matter before the first release.
