# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Environment is managed by pixi (`pyproject.toml` `[tool.pixi.*]`); the package is installed editable into it.

```bash
pixi run test                                          # full suite (pytest tests/ -q; the smoke script is named smoke_*.py so pytest never collects it)
pixi run -e test python -m pytest tests/test_rig.py -q  # one file
pixi run -e test python -m pytest tests/test_rig.py::test_rig_mixed_hw -q   # one test
pixi run smoke-img                                     # tests/smoke_image_formats.py: rgb format/quality comparison (pillow + scikit-image, imgtest env only)
```

No linter/formatter is configured. Without pixi: `pip install -e .[test]` (numpy, opencv-python, h5py).

## Architecture

A small library of per-scene HDF5 writers/readers. Three public classes (`h5scene/__init__.py`, whose docstring is the canonical layout + usage spec — keep it, `README.md`, and `docs/schema.md` in sync when changing the format):

- **`H5Base`** (`h5base.py`) — a *pure file handle*: open/create (`mode` optional: `"r"`/`"r+"` on open, `"w"` on create), context-manager lifecycle, generic `attr`/`set_attr` and `meta`/`set_meta` accessors, `write_section`. Holds no shape or data. Keys are validated against the **subclass's** `ATTR_KEYS` / `META_KEYS` registries (dict name → Python type, `isinstance`-checked on write — numpy scalars accepted for `bool`/`int`/`float`; unknown names raise). `META_KEYS` values are `MetaSpec(type, dtype=…, length="T")` for array keys: dtype (int64) and axis-0 length ("T" = the scene's T) are checked in the base layer — `src_frame_idxs` must be int64 length T, `tags` a `dict[str, str]`. `set_meta` validates BEFORE deleting the old value; `None` removes; both single- and multi-key forms return self. `H5Cam.create/open`, `H5CamRig.build/open` are the public entry points; a failed subclass constructor after create calls `_abort_create()` (handle closed, file removed).
- **`H5Cam`** (`h5cam.py`) — single-camera sequence, shape `(T, H, W)` in `attrs.THW`; `/rgb/{t}` one dataset per frame (rgb is optional). Data lives in the file. `is_metric` defaults False. `set_meta` is the single base form with two call styles: `set_meta(name, value)` (single key; None removes) and `set_meta(dataset=…, scene=…, …)` (multi-key convenience; a kwarg None means "leave as-is") — it is NOT overridden by subclasses. Open validates rgb keys AND section shapes against THW.
- **`H5CamRig`** (`h5camrig.py`) — merges N `H5Cam` files; its `META_KEYS` is **derived** from `H5Cam.META_KEYS` (cam keys minus `seq`/`src_frame_idxs`, + the rig's own `suffix`) into one `(T, N, Hmax, Wmax)` file containing **references only, zero data bytes**: `/rgb/{t}_{n}` are HDF5 ExternalLinks to each camera's `/rgb/{t}` (jpg bytes, never re-encoded; `rgb(t, n)` decodes at that camera's native size), and `/depth /mask /K /T /depth_conf` are HDF5 virtual datasets (VDS) mapping the cameras' compressed data, padded via fillvalue (per-frame `VirtualSource(path, name, (T, h, w))[t]` blocks). Build = validate first, write to a temp file, `os.replace` into place.

Helper package, shared by both subclasses (`h5scene/codecs/` — one code path per section):
- `codecs/sections.py` — the declarative `SECTIONS` table (`Spec`: dtype / compression / compression_opts / shuffle / fillvalue / frame2d / resizable); it drives the H5Cam writers, the rig VDS build, `has()`, and `test_compression_policy`. `has()` is robust to raw/legacy files (no `/rgb` or `/meta` group). `is_metric()` reads attrs first, legacy `/meta/is_metric` as fallback.
- `codecs/depth.py` — meters↔mm (float = meters, **all integer dtypes = mm**), the 0/65535 sentinel rules, the `(h, w)` aspect-ratio + `INTER_NEAREST` resize rules, depth_conf scaling, the derived `depth_mask`.
- `codecs/rgb.py` — jpg q≤95 encoding + DQT-table quality fingerprinting; the **colour-order boundary** (public API is RGB; cv2's BGR is converted at encode/decode); rgb input normalization (arrays / image bytes / file paths); `jpeg_hw()` reads a frame's size from the SOF header alone (the rig's size check avoids full decodes).
- `codecs/pose.py` — K `(3,3)` / T `(4,4)` broadcast.

### Format invariants worth knowing before editing

- **rgb**: stored as jpg bytes with quality ≤ 95 in **RGB channel order** (a red image has a high R channel in the stored jpg and reads back red). Incoming jpg with detected cv2/libjpeg q > 95 or other encodings (PNG, …) is re-encoded to q95; detection only knows the cv2 fingerprint qualities `{1,20,50,70,90,95,99,100}` (a q80 cv2 jpg is "unknown" → re-encoded). Choice justified in `tests/probes/README.md`.
- **depth**: uint16 millimetres; **both 0 and 65535 are invalid** sentinels. Writer maps >= 65.535 m to 0 (never saturates); reader re-clips `>= 65535 -> 0` — 65535 is never a valid depth, so the max representable value cannot round-trip as valid. gzip4 + shuffle. May be stored at smaller `(h, w)` only if `h*W == w*H`; reads resize with `INTER_NEAREST` so sentinels are never interpolated. `depth_conf` (f16) must share depth's `(h, w)`. In the **rig**, depth must be stored at the camera's NATIVE (H, W) — downsampled depth is rejected (a VDS block cannot upscale).
- **mask** bool gzip4 no shuffle; **K / T** stored raw (gzip measured larger for short sequences). `tests/test_compression_policy.py` pins these filter choices.
- **attrs vs /meta** (decision D1): data-side properties (`THW`/`TNHW`, `is_metric`, rig `cam_files`/`cam_hw`/`cam_fp`) are file attrs; dataset-side context goes in `/meta` (`dataset`, `scene`, `seq`, `suffix`, `split`, optional `src_frame_idxs` only when ≠ `arange(T)` — enforced, length must be T, `tags/*` values must be str).
- **Rig rules** (checked at build, before any file is written): same T, `is_metric`, and section set; rgb present in ALL cameras or in NONE; **`dataset` + `scene` must match** (the cameras' seqs merge into one time axis — the rig has **no `seq`** in its registry); exactly one of H/W may differ (the other becomes max, top-left aligned); depth at native (H, W); shared `split`/`tags` inherited only when every camera agrees. Source camera paths stored absolute in `attrs.cam_files` plus a per-camera fingerprint (`T, H, W, size, mtime_ns`); opening a rig raises `FileNotFoundError` (missing) or `ValueError` (rewritten) — a VDS would otherwise silently read fillvalue zeros.

`docs/schema.md` (Chinese) documents the per-class file schema (attrs, sections, /meta keys) for H5Base / H5Cam / H5CamRig; keep it in sync when adding/removing keys.

### Tests

`tests/` uses pytest with `tmp_path` and synthetic data; `tests/probes/` holds real-scene crops (ADT frame) used as inputs. `tests/test_issue_bugs.py` pins the reviewed-bug fixes (per-frame VDS, downsampled rejection, colour round-trip, failed-build preservation, frame-count check, meta no-erase, fingerprint detection, empty/duplicate paths).
