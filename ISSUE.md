# ISSUE.md — h5scene review (2026-11-13)

## 进度表(2026-11-12 19:10 NZST,commit `70d6f43`)

**✅ = 已修(有回归测试)  ⏭️ = 有意跳过(见备注)  每项: 状态 | 处理方式**

| # | 类别 | 状态 | 处理方式 |
|---|------|------|----------|
| 1 | Bug | ✅ | `VirtualSource(path, name, (T, h, w))[t]` 选帧;`test_rig_per_frame_not_frame0` |
| 2 | Bug | ✅ | rig build 拒绝 downsampled depth(VDS 块不能上采样);`test_rig_rejects_downsampled_depth` |
| 3 | Bug | ✅ | `set_rgb` 写时校验帧数 == T;`test_rgb_frame_count_mismatch` |
| 4 | Bug | ✅ | `set_meta` 先校验后删;`test_set_meta_type_error_does_not_erase` |
| 5 | Bug | ✅ | build = 先校验 → 写 temp → `os.replace`;`test_failed_build_preserves_output`(输出=相机路径) |
| 6 | Bug | ✅ | rgb 全有/全无,混合报错;rgb 计入 section 集合 check |
| 7 | Bug | ✅ | `is_metric` 走 `codecs.sections.is_metric`(attrs 优先,legacy 回退),默认 False |
| 8 | Bug | ✅ | 公开 API 全程 RGB;cv2 BGR 在 `codecs.rgb` 边界转换;`test_rgb_colour_roundtrip_from_file` |
| 9 | Bug | ✅ | 所有整数 dtype 当 mm(`codecs.depth.to_mm`) |
| 10 | 文档 | ✅ | 实现文档承诺:mask (H,W)/(T,) 广播、rgb (H,W,3) (T=1)、depth/conf (H,W) (T=1) |
| 11 | 文档 | ✅ | 新 `.create()/.open()/.build()` classmethod;open 默认 `"r"`,README 改写 |
| 12 | 文档 | ✅ | 决定:全部相机一致时继承 split/tags/meta_json(不一致不写) |
| 13 | 文档 | ✅ | `frame_indices` 强制 length T;`== arange(T)` 时不存(删) |
| 14 | 文档 | ✅ | docstring 改准:DQT 指纹只认 cv2 8 档 q,q80 = unknown → 重编码 q95 |
| 15 | 文档 | ✅ | `_NUM_EQUIV`:np.bool_/np.integer/np.floating 对应 bool/int/float |
| 16 | 文档 | ✅ | 全部过时 docstring 清掉(随模块拆分重写) |
| 17 | 文档 | ✅ | 删 `_io.py`/`rgb.py`/`convert.py`(→ `_tobedelete/`);新 `codecs/{rgb,depth,pose,sections}`,单一代码路径 |
| 18 | 健壮 | ✅ | 空/重复 cam_paths 明确报错;`test_rig_rejects_empty_and_duplicate_paths` |
| 19 | 健壮 | ✅ | rgb 尺寸 check 走 `jpeg_hw()`(SOF header,不全解码) |
| 20 | 健壮 | ✅ | `attrs.cam_fp` 指纹(T,H,W,size,mtime_ns);open 校验;`test_rig_open_detects_rewritten_camera` |
| 21 | 健壮 | ✅ | `H5Cam.open` 校验 section 形状 vs THW(`_check_sections`) |
| 22 | 健壮 | ✅ | `has()` 用 `h.get` 容忍缺 /rgb 或 /meta(老文件) |
| 23 | 健壮 | ✅ | `_abort_create()`:子类 create 后失败 → 关句柄 + 删半成品文件 |
| 24 | 健壮 | ✅ | `set_meta` dict 值必须 str(不再静默 str()) |
| 25 | 测试 | ✅ | `tests/test_issue_bugs.py` 8 项(含 multi-t rig、downsampled、色彩、failed-build、帧数、fingerprint);`sys.path.insert` 全部移除 |
| 26 | 卫生 | ✅ | `c0.h5/c1.h5` → `_tobedelete/`(git rm);`*.h5` gitignore(probes 除外);pillow/scikit-image 只留 `imgtest` feature,`smoke-img` 任务改 `-e imgtest` |
| 27 | 结构 | ✅ | 模块按角色拆分:`codecs/{rgb,depth,pose,sections}` + `h5base/h5cam/h5camrig`(保留 h5 前缀文件名) |
| 28 | 结构 | ✅ | 声明式 `SECTIONS` Spec 表驱动 H5Cam writer / rig VDS build / `has()` / `test_compression_policy` |
| 29 | 结构 | ⏭️ | reader mixin 未做(随 #33 的 `_RigReaders` 一起退回) |
| 30 | 结构 | ✅ | `.create()/.open()/.build()` classmethod 已加(11 项);`__init__` 仍是入口(兼容),classmethod 是推荐路径 |
| 31 | 结构 | ✅ | `H5Cam.set_meta` 遮蔽删掉:基类 `set_meta` 一个方法两种形式——`set_meta(name, value)`(单键,None=删)和 `set_meta(**kw)`(多键,None=不动);子类语义走 `_meta_value` 钩子(frame_indices dense→不存);`H5Cam.set_meta is H5Base.set_meta`;`test_set_meta_is_not_overridden` |
| 32 | 结构 | ✅ | `MetaSpec(type, dtype=, length="T")` 注册表:h5base 校验 dtype+axis-0 长度(frame_indices 的 (T,) 在基类层强拒,#13 的 length-7 不再接受);rig 的 `META_KEYS` 从 `H5Cam.META_KEYS` 推导(cam keys minus seq/frame_indices + suffix);`test_frame_indices_length_enforced` |
| 33 | 结构 | ⏭️ | in-memory `CamRig` 做过后**按用户要求退回**(未提交):内存版每帧读多一次 pad 拷贝、多一个持有句柄的类,收益不大;VDS 文件版保留为唯一形态 |
| 34 | 结构 | ✅ | `__len__` / `sections` / `__repr__`(两个类都有) |
| 35 | 结构 | ✅ | 测试镜像模块:`tests/test_codecs_depth.py`(7)+ `tests/test_codecs_rgb.py`(10,纯 numpy/cv2,imgtest env 不需要)+ `test_h5cam.py`→`test_cam.py`(git mv);`test_rig.py` 旧 docstring 修掉;pixi `test` 任务去掉 `--ignore`(smoke 按 `smoke_*.py` 命名本就不被 pytest 收集——已验证 collect-only 为 0)。**注:smoke 文件没搬去 scripts/**——用户的 cleanliness 规则强制 tests/smoke 脚本必须在 tests/,与 ISSUE 建议冲突,以用户规则为准 |

**计数:✅ 33(+ 结构 #35 测试布局)/ ⏭️ 2(#29 reader mixin · #33 in-memory CamRig(做过后退回))**

## 追加:OCR 独立审计(18 条,2026-11-12 20:27 NZST)
**✅ 修了 8 项(各带回归测试或 diff):rig build 的 cams 句柄泄漏(try/finally)· h5cam open 只捕 ValueError(改 Exception)· `_abort_create` close 无保护 + `__exit__` close 掩盖原异常 · to_mm 恰好 65.535 m 可写成 65535(→ 0,读写对哨兵达成一致)· normalize_rgb 空输入 `(None,None,())`(→ 明确报错)· encode_rgb 不查 imencode success · 死代码 ×3(rig 的 filter_args / h5cam 的 os / rig_scan 的 eager-load hw)· requires-python `>=3.10` 与 pillow13/skimage0.26 的 `>=3.11` 冲突 → `>=3.11`**
**❌ 误报 3 项:`Image.NEAREST` ×2 称"pillow13 已删、critical 崩溃"——实测 pillow 13.0.0 下 `Image.NEAREST == 0` 仍在(smoke 一直跑通);h5cam 的 `filter_args` 称"未用"——实际 3 处 `filter_args("depth")` 在用**
**⏭️ 有意不修:`set_rgb` 非原子替换(先删后写,中途失败留半组)——已加代码注记,重构成 temp-name+rename 超出本库体量;`normalize_rgb`/`to_mm` 的 T 参数未用(为与 mask/pose 的签名对称,帧数由 H5Cam 层校验)****

## 进度表说明(2026-11-13 review 前)

Items marked **[verified]** were reproduced with a scratch script. Line numbers refer to the current `master`.

Items marked **[verified]** were reproduced with a scratch script. Line numbers refer to the current `master`.

## Bugs

1. **[verified] Rig VDS per-frame mapping always reads frame 0** — `h5camrig.py:266-269`.
   `h5py.VirtualSource(path, name, shape=(h_, w_))` declares a 2-D source for a 3-D `(T, h, w)` dataset and never selects index `t`, so every `t` maps to camera frame 0. Any camera that takes the per-frame branch (smaller H/W, or downsampled depth) returns frame-0 depth/mask/depth_conf for all t.
   Repro: depth = 1,2,3 m per t → `rig.depth(t, small_cam)` gives `[1000, 1000, 1000]`. The fix is `VirtualSource(path, name, shape=(T, h_, w_))[t]`.
   `test_rig_mixed_hw` only checks `t=0`, so it misses this.

2. **[verified] Downsampled depth is not upscaled in the rig** — `h5camrig.py:262-269`, `135`.
   A camera whose depth is stored at `(h, w) < (H, W)` (an allowed format) is placed unscaled in the top-left of `(Hmax, Wmax)`, and the rest is filled with 0. The rig read never resizes, because the VDS frame is already `(Hmax, Wmax)`. Repro: an 8×8 camera with 4×4 depth gives 64 valid pixels via `H5Cam` but only 16 via the rig, and they are in the wrong place. The rig must either reject downsampled depth or map it at the camera's own size.

3. **[verified] `H5Cam.set_rgb` does not check the frame count against T** — `h5cam.py:123-133`. Writing 2 frames into a T=3 file succeeds, and then reopening the file fails with `rgb keys != 0..T-1`.

4. **[verified] `H5Base.set_meta` deletes the old value before the type check** — `h5base.py:98-104`. `set_meta("scene", 5)` raises `TypeError` but has already erased the existing `scene`. Validate first, then delete.

5. **[verified] A failed rig build deletes an existing output file** — `h5camrig.py:194-197`. `os.remove(out_path)` runs before any validation. If `out_path` is one of the camera paths, that camera file is deleted. A write error also leaves a partial rig file behind. Validate first, write to a temp file, then rename.

6. **[verified] A rig requires rgb, but rgb is optional in H5Cam** — `h5camrig.py:226`. Cameras without rgb make the build crash with a raw `KeyError`. Either skip rgb when it is absent from all cameras, or give a clear error. rgb should also be part of the section-set equality check.

7. **[verified] Broken `is_metric` fallback in the rig** — `h5camrig.py:101-107`. `self.meta("is_metric")` always raises `ValueError`, because the key is not in `META_KEYS`. The fallback default is `True`, while `H5Cam` defaults to `False`. Use `_io.is_metric` like `H5Cam` does.

8. **[verified] Colour order: BGR is stored and returned while the docs say RGB** — `_io.decode_rgb`, `rgb._encode_jpg`.
   - cv2 encodes arrays as BGR, and `decode_rgb` returns BGR.
   - Arrays round-trip unchanged, but the stored jpgs have swapped channels, so other tools see wrong colours.
   - Images supplied as bytes or paths come back as BGR, so a red PNG reads back as `[0, 0, 254]`.
   - The result depends on the input kind. Convert RGB↔BGR at encode and decode.

9. **[verified] Integer depth that isn't uint16 is treated as metres** — `convert.py:81`. An `int32` array of mm values (1500) is multiplied by 1000, goes past 65535, and every pixel is silently stored as invalid (0). Treat all integer dtypes as mm, or reject them.

## Docs vs code inconsistencies

10. **[verified]** The `H5Cam` docstrings and README promise inputs that `H5Cam._cam` and `_rgb_frames` reject when T>1:
    - mask `(H, W)` broadcast
    - mask `(T,)` per-frame slots
    - rgb `(H, W, 3)` broadcast
    - depth/depth_conf `(H, W)` broadcast

    Either implement them or fix the docs (`h5cam.py:259, 310, 327`).
11. **[verified]** The README shows `with H5Cam("cam.h5") as cam: cam.set_attr(...)` / `cam.set_meta(...)`, but open mode is read-only (`"r"`), so these raise. The same applies to the rig: it declares `split/meta_json/frame_indices/tags` in `META_KEYS`, but the build never writes them and the opened file can't be written. Add an `mode="r+"` option, or drop those keys from the rig and its usage docs.
12. The rig drops the cameras' `split`, `tags`, `meta_json` and `is_metric` consistency info. Only `dataset`, `scene` and `suffix` are written (`h5camrig.py:271-274`). Decide whether the shared `split`/`tags` should be inherited.
13. The docs say `frame_indices` is written "only when ≠ arange(T)" and is `(T,)`. Neither is enforced: a length of 7 is accepted for T=3 [verified] (`h5cam.py:253`, `h5base.py:109-112`).
14. The `rgb._load_rgb_frames` docstring says jpgs with "detected quality ≤ 95" are stored verbatim. Detection only knows q ∈ {1, 20, 50, 70, 90, 95, 99, 100} from cv2, so a cv2 q80 jpg is "unknown" [verified] and gets re-encoded at q95: a larger file plus generation loss. Every non-cv2 jpg is re-encoded too. Either estimate the quality from the DQT scale, or reword the docs.
15. **[verified]** `set_attr("is_metric", np.bool_(True))` raises `TypeError … expects bool, got bool`, because `np.bool_` is not a `bool`. The same applies to numpy scalars in general. Accept numpy equivalents.
16. Stale docstrings:
    - The `h5base.py` header shows the registries as tuples of pairs, but they are dicts.
    - The `h5base.py:37` comment says a "str label = descriptive only", but none are used.
    - The `tests/test_rig.py` header mentions `h5scene/rig.py` and a "/rgb physically" copy.
    - The `_io.py` docstring says the write helpers are shared by both classes, but neither class uses them (see 17).
    - `H5Cam.set_meta`'s docstring points to `H5Base.META_KEYS`, which is empty.
17. Dead code in `_io.py`, left over from the old H5Scene: `write_rgb`, `write_depth`, `write_depth_conf`, `write_mask`, `write_K`, `write_pose` and `_check_depth_hw` are unused. So are `rgb._normalize_rgb` and `convert._frame_keys`, which are only reached through them. `H5Cam` reimplements the same logic: `_check_depth_hw` duplicates `_io._check_depth_hw`, and `_rgb_frames` duplicates `_normalize_rgb`. Keep one copy.

## Robustness / improvements

18. **[verified]** `H5CamRig(out, [])` gives a confusing `camera T mismatch: []`. Reject an empty or duplicate `cam_paths` explicitly.
19. Rig build decodes every rgb frame of every camera (T×N jpeg decodes) only to check sizes (`h5camrig.py:224-229`). Reading the jpg SOF header, or checking frame 0, would be much cheaper.
20. Opening a rig checks only that the camera files exist. It does not check that they still match: if a camera is rewritten with a different T, H/W or section set, the VDS silently misreads. Store a cheap fingerprint per camera (THW, mtime/size) and verify it on open.
21. `H5Cam` open validates only the rgb keys. It does not check that the section shapes (`depth`, `mask`, `K`, `T`) match `THW`.
22. `_io.has()` raises `KeyError` when the file has no `/rgb` or `/meta` group (raw or legacy files). Its `"meta_json" in g` check looks like a leftover. Use `.get`.
23. `H5Base(create=True)` leaves an open handle and a half-made file if the subclass constructor then fails (e.g. a bad T/H/W). Close the handle and remove the file on error.
24. `set_meta(tags=…)` silently `str()`s non-string values, including nested dicts. Validate `dict[str, str]`.
25. Tests:
    - Add multi-t checks for the rig (would catch 1).
    - Add a rig built from downsampled depth (2).
    - Add a colour round-trip from file input (8).
    - Add a failed-build-preserves-output test (5).
    - Add an rgb frame-count mismatch test (3).
    - The tests use `sys.path.insert` even though the package is installed editable; that can be dropped.
26. Repo hygiene:
    - `c0.h5` and `c1.h5` are committed at the repo root and look like leftover test outputs. Remove them and add `*.h5` (except probes) to `.gitignore`.
    - In `pyproject.toml`, pillow and scikit-image are listed both in the default pixi deps and in the `imgtest` feature. Keep only the feature, so the default env stays minimal.

## Structure / design suggestions

27. **Split the modules by role, not by history.** Today the rgb policy lives in `rgb.py` and is partly duplicated in `h5cam.py`, and the depth rules are spread over `convert.py`, `_io.py` and `h5cam.py`. A clearer layout:
    ```
    h5scene/
      base.py      H5Base (handle, registries, open modes)
      cam.py       H5Cam
      rig.py       H5CamRig
      codecs/
        rgb.py     encode/decode/quality (the only place cv2 colour order lives)
        depth.py   to_mm / to_meters / resize / sentinels / depth_mask
        mask.py, pose.py (or one sections.py)
      schema.py    section specs + meta key registries, in one table
    ```
    Delete the dead `_io` write helpers (17) so there is one code path per section.
28. **One declarative section table.** dtype, frame shape, compression/chunks, fillvalue and the read decoder are hard-coded separately in `H5Cam.set_*`, `_io` and the rig's `_SECTIONS`. A single `SECTIONS = {name: Spec(dtype, compression, shuffle, fill, resizable, decoder)}` would drive:
    - the H5Cam writers
    - the rig's VDS build
    - `has()`
    - `test_compression_policy`

    The rules then can't drift apart.
29. **Share the reader API through a mixin.** `H5Cam` and `H5CamRig` duplicate the `dataset/scene/suffix/split/meta_json/frame_indices/tags` properties, `has`, `is_metric`, `depth/depth_conf/mask/K/pose/depth_mask`. They differ only in the frame index (`[t]` vs `[t, n]`). Implement those once in a reader mixin over an abstract `_frame(name, *idx)`, and keep only shape and indexing in the subclasses.
30. **Explicit open modes.** Replace "constructor args decide create vs open" with classmethods:
    - `H5Cam.create(path, T, H, W)`
    - `H5Cam.open(path, mode="r"|"r+")`
    - `H5CamRig.build(out, cams, suffix)`
    - `H5CamRig.open(path)`

    This removes the overloaded `__init__` and the `self.path`/`_h` setup duplicated in `_build`, which bypasses `H5Base.__init__`. It also fixes 11.
31. **Don't shadow `set_meta` with a different signature.** `H5Cam.set_meta(**kw)` versus `H5Base.set_meta(name, value)` breaks substitutability and forces the `H5Base.set_meta(cam, …)` workaround. Rename the convenience method to `update_meta(**kw)`, or make the base accept both forms.
32. **Registries as typed specs.** Use `META_KEYS = {"frame_indices": Spec(np.ndarray, dtype=int64, shape=("T",))}` so that shape, dtype and numpy-scalar acceptance are checked in one place (13, 15). The rig could then derive its registry as "cam keys minus seq".
33. **Rig as a lazy view, not only a file.** The rig's value is the `(T, N)` indexing. An in-memory `CamRig([cam paths])` reader, with the same API and no file, would cover most uses. The VDS file stays an optional persisted form. This also avoids the stale-reference problem (20).
34. **Small API polish.**
    - Batch reads: `depth(slice)` or `depth()` for all frames. Today it is per-frame only, with one h5 read per call.
    - `__len__` → T.
    - `__repr__` showing the shape and the sections present.
    - `sections` property instead of calling `has()` per name.
35. **Tests layout.** Mirror the modules: `test_codecs_rgb.py`, `test_codecs_depth.py`, `test_cam.py`, `test_rig.py`. Move `smoke_image_formats.py` and `probes/rig_scan.py` to a `scripts/` or `bench/` directory so pytest discovery is unambiguous and the `--ignore` in the pixi task isn't needed.
