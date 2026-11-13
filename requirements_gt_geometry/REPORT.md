# 交付报告：H5Geo（GT 几何）压缩实验 + Phase 1 框架

日期：2026-11-13。对应 `REQUIREMENTS_GT_GEOMETRY.md` 与 `IMPLEMENTATION_SPEC.md`。
**结论先行：spec 的"压缩实验"（§5.0–5.3）已全部跑完并定稿；Phase 1 代码框架（§10 第 1–4 步）已交付，全套测试 90 passed。尚未交付：Phase 2（多边形/detail/材质）与 §5.4 的 43 场景验收（在 SpatialBench 侧）。**

## 一、压缩实验结果（§5.0–5.3）

### 1. 全矩阵 sweep（§5.2）— `tests/probes/geo/README.md`

- 脚本：`tests/smoke_geo_compression.py`（自带 ~30 行 binary-ply parser，不进库，D7），`pixi run smoke-geo [ply 目录]`，`--write-probes` 可重生成探针。
- 数据：5 个 sample ply（`sample_plys/` 目录，实验后已归档到 `_tobedelete/requirements_gt_geometry/`；数据事实与 per-source 尺寸表见 REQUIREMENTS_GT_GEOMETRY.md），每 source 取 50k 点探针（scannetpp 取只用这些点的子网格重编号）；矩阵 = P/Cd/N/ptnum × {none, gzip1/4/6/9 ± sh, lzf ± sh} × chunk 16k/64k/256k × ptnum flat (V,) vs (F,3)，共 468 格；每格测存字节、对 raw 比值、全读 `ds[:]` 时间、写时间、逐位回环。
- **全局检查（§5.0 第 6 条）：43 场景投影合计 811 MB < 1.8 GB，通过**（逐 source 比值 × 需求表 per-source 字节，见报告表）。
- 逐属性按 §5.0 规则：全部 lossless；5% 带宽内 P 最优 gzip6+sh@256k（41.89 MB）、N gzip6+sh@256k、ptnum gzip1+sh@64k，读速全部在 2×C4 之上故走了 fallback（规则第 4 步"取最快读"）。
- **C6（ptnum 布局）**：flat (V,) 与 (F,3) 存字节基本相等（3.20 vs 3.18 MB）→ 选 flat (V,)。
- **C7（per-source 例外）**：7scenes 的 P 用纯 gzip 只省投影 10.2 MB < 阈值（43 场景合计的 10% = 81.1 MB）→ **不采纳**，不做 per-source 策略。

### 2. hdf5plugin 测试（zstd/blosc/lz4）— `tests/probes/hdf5plugin/README.md`

- 脚本：`tests/smoke_hdf5plugin_compression.py`，`pixi run smoke-plugin`（pixi `plugin` env，hdf5plugin 7.x；不污染默认 env）。
- 协议同 geo sweep（存字节、比值、全读、逐位回环；点云探针 + 真实 640×640 uint16 深度帧）。
- **结论：zstd/blosc/lz4 全面不赢 gzip**——13 组里只有 7scenes P 的 zstd9 略小（0.79×，读速持平），其余 ≥1.0× 或更慢；深度帧上 gzip4+sh 最小最快（zstd9 1.10×）。支持 spec §5.0 把 hdf5plugin 判为 *not eligible (new dependency)* 的决定。

### 3. 定稿的 `GEO_ATTRS`（owner 决策，已记录在 `tests/probes/geo/README.md` 开头）

| 属性 | 存 dtype | filter | chunk rows | 依据 |
|---|---|---|---|---|
| P | (M,3) f32 | **gzip4 + shuffle** | 262144 | owner：沿用 /depth 的库约定；sweep 测出 gzip6+sh 仅快 ~9%、小 0.3 MB，不值得破坏一致性 |
| N | (M,3) f32 | **gzip4 + shuffle** | 262144 | 同上 |
| Cd | (M,3) u8 | **gzip4，无 shuffle** | 262144 | 1 字节：shuffle 无字节可拆（同 mask 的推理），C1 存活 |
| ptnum | (V,) i32 | **gzip4 + shuffle** | 65536 | owner 同上；C6 确认 flat 布局 |
| 其余行 | — | 多字节 gzip4+sh / 1 字节 gzip4 | 65536 | 按同一规则外推（未单独 sweep） |

## 二、Phase 1 代码交付（§10 第 1–4 步）

- **`h5scene/h5geo.py`**（新）：`H5Geo.create(P, is_metric=False)` 钉 `attrs.n_points`；`open` 时 O(1) 校验（shape/dtype/长度规则：/points 轴0=M、/vertices=V、/prims=F；ptnum 值域读时才查，D5）；三角拓扑按一个单元写（`set_triangles`/`triangles`/`del_topology` + `prim_size=3`，V/F 派生不存）；point/vertex/prim 三级通用 accessor（标准名查 GEO_ATTRS，自定义名走 dtype 白名单 D10 + 长度规则）；P/Cd/N 快捷键；`bbox`；meta = H5Cam 键减 seq/src_frame_idxs；create 失败删半成品文件（同 H5Cam 约定）。
- **`h5scene/codecs/sections.py`**：`GEO_ATTRS`（20 行，上表定稿）、`GeoSpec`（含 chunk_rows）、`geo_chunks`、`CUSTOM_DTYPES`；`filter_args`/`has` 加 `table=` 参数（默认 SECTIONS，H5Cam/Rig 现有调用零改动）。
- **测试**：
  - `tests/test_geo.py`（11 个）：mesh/点云 round-trip、f64→f32、float Cd 拒绝、坏三角索引、raw-h5py 写坏 ptnum 读时拒绝（D5）、缺 P/缺根属性 open 抛错且句柄关闭、长度规则（写时拒绝 + 手改后 open 拒绝）、未知数据集拒绝 / 自定义属性允许、失败写不伤旧数据、create 失败删文件、meta 键边界。
  - `tests/test_compression_policy.py` 新增 2 个 on-disk filter 断言（§5.3）：`test_geo_P_N_ptnum_gzip_shuffle`、`test_geo_Cd_gzip_no_shuffle`（docstring 引 §5 数字 + owner 决策）。
  - 探针 `.npy` 已落盘 `tests/probes/geo/<source>/`（5 source，~8 MB；scannetpp 含 ptnum，7scenes 含 N）。
- **测试：90 passed**（原 77 + 新 13；其中 geo 相关 `test_geo.py` + `test_compression_policy.py` + `test_codecs_rgb.py` 37 passed，2026-11-13 15:24 NZDT 实测）。

## 三、需求方需要知道的未完成项

1. **§5.4 全 43 场景验收未跑**：需要在 SpatialBench 侧用 open3d 转换器（D7，不在本库）跑 43 场景，按 §5.4 的 pass 条件（总 <1.8 GB、P atol 1e-6 / Cd/N/ptnum 精确、面数一致）出数并回填 `tests/probes/geo/README.md`。本库侧的接收路径（H5Geo.create + set_*）已就绪。
2. **Phase 2 未实现**（§10 第 6–11 步）：quad/多边形（prim_size 0 + vertexcount 的写路径）、detail 命名表（class/material 名）、纹理（deferred）、groups、normal helpers。schema 已按两阶段共享一份设计（D13），Phase 1 的通用 accessor 与长度校验可直接复用。
3. **§9 文档同步未完成**：`h5scene/__init__.py` docstring、`docs/schema.md` 的 H5Geo 节、`README.md` 用法、`CLAUDE.md` 不变量——本次只交付了 `codecs/sections.py` 与 `h5scene/h5geo.py` 内自带 docstring，外层文档待补。
4. 探针 `.npy`（~8 MB）尚未提交 git，是否入库请定。

## 四、复现命令

```sh
pixi run smoke-geo                 # geo sweep（默认 sample_plys；--write-probes 重生成探针）
pixi run smoke-plugin              # hdf5plugin 测试（plugin env）
pixi run test                      # 全套 pytest（含 test_geo.py 与 §5.3 断言）
```
