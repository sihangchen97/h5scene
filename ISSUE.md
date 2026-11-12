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

## Closed: review passes 5–6 #71–#81 (2026-11-13)

- **Rig fingerprint (#71, #79):** removed. `cam_fp` is gone; open compares each camera's `attrs.THW` with `(T, *cam_hw[n])` and checks every mapped section. A metadata edit on a camera no longer breaks the rig, because a VDS is a live reference.
- **Input guards:**
  - `depth_hw_ok` requires `h, w >= 1` (#74).
  - `H5Cam.create` rejects non-integral or `bool` T/H/W with a `TypeError` (#75, #80).
  - `open` requires `attrs.is_metric` on both classes (#76).
- **Docs, text and tests:** stale per-frame-VDS and fingerprint wording removed, plus small leftovers (#72, #73, #78, #81). Regression tests were added for each fix.
- ⏭️ #77: `_tobedelete/` is the gitignored local mv-not-rm holding area, not repo content.

## Open

None.
