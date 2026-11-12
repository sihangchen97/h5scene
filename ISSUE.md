# ISSUE.md — h5scene review

✅ fixed · ⏭️ dropped on purpose · ❌ false positive · 🔲 open
**[verified]** = reproduced with a scratch script.

## Closed: pre-release review #1–#70 (2026-11-12 → 2026-11-13, up to `67d7aba`)

Four review passes plus an independent audit. Everything is fixed, dropped on purpose, or was a false positive. Regression tests for the fixes are in `tests/test_issue_bugs.py`; the per-pass tables are in git history (`git log -p ISSUE.md`).

- **Rig correctness:** VDS selects the right frame, with one mapping per camera (#1, #54). Build validates, writes a temp file, then `os.replace`; a failed build leaves no partial file and no empty dir (#5, #42, #46, #47). It rejects:
  - downsampled depth (#2);
  - rgb in only some cameras (#6);
  - empty or duplicate camera paths (#18);
  - an output path that is one of the cameras (#36).
- **Rig sources:** `cam_files` are relative to the rig and the content fingerprint is `(T, H, W, size)`, which lets a rig survive `cp` or a move (#20, #39, #40). Open also checks that every mapped section and `/img` frame is still there with the mapped shape (#51, #57). Only the 4-column `cam_fp` is accepted, and `cam_fp`/`cam_hw` are required (#60, #61).
- **Formats:**
  - **rgb:** public API is RGB (#8). A jpg is stored only if its quality is ≤ 95 (#14, #45). `jpeg_hw` reads the SOF header and skips the DHT/JPG/DAC markers (#19, #44, #50).
  - **depth:** explicit-unit writers (#9). 65.535 m is written as 0 (audit). `depth_conf` accepts 0-1 floats only (#41).
- **Meta and attrs:**
  - `set_meta` validates every key before it writes, and mixing its two call styles raises (#4, #37, #38). It is defined once, in the base class (#31).
  - The base class checks `MetaSpec`; the rig's registry is derived from the camera's (#32, #48).
  - `is_metric` is read only from attrs (#62). `has()` answers sections and img only (#63).
- **API:**
  - `create`/`open`/`build` are the only constructors (#11, #30, #65).
  - `/rgb` was renamed to `/img`, and `set_depth_mm`/`set_depth_meters`/`set_depth_conf_01`/`set_pose_c2w` and `bgr` were added. Nothing was released, so this needs no migration.
  - `del_section` was added (#43, #52).
- **Structure and cleanup:**
  - Code is split into `codecs/*` and `h5base`/`h5cam`/`h5camrig` (#27). One `SECTIONS` table drives the writers, VDS, `has()` and the compression test (#28).
  - Dead code, stale text and unused `T` parameters were removed (#16, #17, #49, #55, #59, #66, #67). The legacy layout branches in `rig_scan.py` are gone (#68, #69). The pixi `test` task now runs in the test env (#53).
- **Dropped on purpose:**
  - the reader mixin and the in-memory `CamRig` (#29, #33);
  - batch reads (#34);
  - an atomic `set_rgb` replace (audit).
- **False positives:** `Image.NEAREST` ×2 and "`filter_args` unused" (audit).

## Review pass 5 (2026-11-13, at `67d7aba`; fixed in `1b77006`)

71. ✅ **`cam_fp` no longer stores the file size** — it is `(N, 3) = (T, H, W)` (the camera's own THW, re-read from the camera at open); a metadata edit no longer breaks open, the size check only produced false alarms. `docs/schema.md` + `h5scene/__init__.py` updated; test_camera_meta_edit_does_not_break_rig_open.
    - **Repro:** run `H5Cam.open(a, "r+").set_meta("split", "train")` after the build. `H5CamRig.open` then raises "no longer matches the fingerprint". The edit grows the HDF5 file, and the fingerprint includes the file size.
    - **The size doesn't catch real changes either:** rewriting depth at the same shape keeps the size, and the rig opens and reads the new values. That is correct, because a VDS is a live reference.
    - **Why it can go:** since #51, open checks each mapped section's presence and shape, which covers everything the size check was for. The size now only produces false alarms.
    - **Fix:** drop `size` from `cam_fp` (it becomes `(T, H, W)`, which is redundant with `cam_hw` plus T). Alternatively, keep it but document that any camera write invalidates the rig. Update `h5camrig.py` docstring L25-34 and `docs/schema.md`.
72. ✅ **Stale "per-frame VDS blocks" text removed** (h5camrig docstring + error message, test_issue_bugs, CLAUDE.md).
    - `h5camrig.py` L21-24 (module docstring) and L363 (error message: "the rig maps per-frame VDS blocks").
    - `tests/test_issue_bugs.py` L30-33 ("the per-frame branch runs").
    - CLAUDE.md L44 ("per-frame VDS").
73. ✅ **All leftovers fixed** (MetaSpec docstring example, trailing commas, `ds` reuse, the jpeg comment, `cam._created = True`, double blank line).
    - The `MetaSpec`/`H5Base` docstring example still derives with `if k != "seq"` (`h5base.py` L45, L102). The rig also drops `src_frame_idxs`.
    - The #66 edit left trailing commas: `def to_mm(mm,)`, `to_meters_mm(meters,)`, `to_conf(conf,)`.
    - In `_validate`, `ds` is reused, first for the dataset set and then for the img dataset (L332 vs L367).
    - The comment "not a jpeg at all" in `normalize_rgb` (`rgb.py` L281) is wrong: `q` is not None there, so it is a jpeg without a SOF.
    - `H5Cam.create` sets `cam._created = True` again, but the base `__init__` already sets it.
    - `h5camrig.py` L49-50 has a double blank line.
74. ✅ **`(T, 0, 0)` depth is now a clear ValueError** — `depth_hw_ok` requires `h, w >= 1` (test_depth_zero_hw_is_rejected_cleanly). with "All chunk dimensions must be positive". `depth_hw_ok` accepts `h = w = 0` because `0*W == 0*H`. Fix: require `h, w >= 1` in `depth_hw_ok`.
75. ✅ **`create` rejects non-integral T/H/W** via `operator.index` (test_create_rejects_non_integral_thw). Any number ≥ 1 is accepted and then truncated by `int()`. Fix: require integral values (`operator.index`, or `int(x) == x`).
76. ✅ **`open` requires `attrs.is_metric`** on both H5Cam and H5CamRig (test_open_requires_is_metric). (#62 said it could be required). A foreign file with only `THW` opens, and the `is_metric` property raises `KeyError` later. Fix: add `is_metric` to the "not an H5Cam/H5CamRig file" checks.
77. ⏭️ **`_tobedelete/probe_t.h5` is still on disk**, although #69 says the directory was deleted. It is gitignored and is the local mv-not-rm holding area, so this is a wording mismatch only. Remove it by hand when convenient.

## Open: review of the #71–#76 fixes (2026-11-13, at `1b77006`)

78. ✅ **README.md updated for #71 (and #79).** L46 still says `cam_fp = int64 (N, 4) per-camera (T, H, W, size)`, and L81 says "content fingerprint (THW + size)".
79. ✅ **`cam_fp` dropped**: open compares the camera THW with `(T, *cam_hw[n])`. Was: `cam_fp` is now fully redundant. Each row is `(TNHW[0], *cam_hw[n])`, and `_verify_sources` even asserts that equality. It is also not a "content" fingerprint (see CLAUDE.md L38, `__init__.py` L85 and `schema.md`). Pick one:
    - drop `cam_fp` and compare the camera's THW with `(T, *cam_hw[n])` directly;
    - or keep it and call it "the camera THW at build time", not a content fingerprint.
80. ✅ **`H5Cam.create` type handling fixed.** Non-integral T/H/W now raises `TypeError` (`from None`); `bool` is rejected explicitly (it would have passed as 1 via `operator.index(True) == 1`).
81. ✅ **Tests updated for #79.** The `cam_fp` assertion is gone (now asserts `cam_hw`), the foreign-rig fixture no longer sets `cam_fp`, and the stale "fingerprint (THW + size)" / "fingerprint is content-based" / "fingerprint detection" comments are reworded.
    - **Fails:** `tests/test_issue_bugs.py` L287 asserts `h.attrs["cam_fp"].shape == (2, 3)` → `test_rig_survives_cp_and_moves_with_cameras` raises `KeyError`. Fix: assert `"cam_fp" not in h.attrs` and `cam_hw.shape == (2, 2)`.
    - **Stale text:**
      - L107: "fingerprint (THW + size)".
      - L288: "fingerprint is content-based".
      - L345: the foreign-rig fixture still sets `cam_fp`.
      - L352: the #71 docstring says "fingerprint".
      - CLAUDE.md L44: "fingerprint detection".
      - `h5camrig.py` L59: the class docstring says "match the stored fingerprint".
