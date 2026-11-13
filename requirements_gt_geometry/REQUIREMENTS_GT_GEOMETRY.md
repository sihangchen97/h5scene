# 需求：GT 几何（网格/点云）类 — 来自 SpatialBench（2026-11-13）

## 背景

SpatialBench 的 point-cloud 指标（DA3 TSDF 口径）需要一个 GT 几何：每场景一个
.ply，共 5 个 source、43 个场景、1.8 GB（`dataset/pointcloud/<source>/<scene>/*.ply`）。
现状评测直接 `open3d.io.read_triangle_mesh(ply)`，本需求希望把这些 GT 也放进
h5scene，与 sparse_h5 的 H5Cam 同库同风格（一个场景一个 h5，可被
`H5CamRig` 之类的上层结构引用/打包）。

> 更正（2026-11-13 全量转换实测）：ply 实为 **45** 个（下文 43 系早期数错）。

## 数据事实（已实测全部 45 个 ply）

| source | 场景数 | 类型 | 顶点 | 面 | 坐标精度 | 颜色 | 法向 | 单文件大小 |
|---|---|---|---|---|---|---|---|---|
| scannetpp | 11 | **三角网格** | 0.5M–3.4M | 1.5M–6.9M | **float32** m | uchar RGB | 无 | 14–142 MB |
| dtu | 13 | 点云（无面） | 1M 固定 | 无 | **double** m | uchar RGB | 无 | 27 MB |
| nrgbd | 8 | 点云 | 1M 固定 | 无 | double m | uchar RGB | 无 | ~27 MB |
| 7scenes | 7 | 点云 | 1M 固定 | 无 | double m | uchar RGB | **有**（double ×3） | 51 MB |
| hiroom | 6 | 点云 | 1.1M–1.2M | 无 | double m | uchar RGB | 无 | 38 MB |

- 全部为 world 坐标、米制（DA3-aligned，与 H5Cam 的 /T 同一世界系）；每个 source
  的 ply 在 scene 目录下**唯一**（评测侧 `get_gt_mesh_path` 靠 glob 取第一个）。
- 点云并非全 1M：hiroom 0.33M–1.2M，dtu 0.46M–1M，7scenes heads_seq-02 0.51M。
- ply 头信息（binary_little_endian 1.0）：scannetpp 面为 `property list uchar int
  vertex_indices`；其余无 `element face`。

## 功能需求（最小集）

1. 新类，存**静态世界系几何**（没有时间轴，不与 H5Cam 的 THW 体系耦合）：
   - `/vertices` — (M, 3) float32，米制，**必有**；
   - `/faces` — (F, 3) int32 顶点索引，**可缺省**（缺省 = 点云，M 个点）；
   - `/colors` — (M, 3) uint8 RGB，可缺省；
   - `/normals` — (M, 3) float32，可缺省（仅 7scenes GT 有）。
2. 读写 API 风格对齐 H5Cam：
   - `cls.create(path, n_vertices[, n_faces])`（或 create 时不预声明、`set_*` 定形
     —— 由你们定，倾向 set_* 定形，因为点云/网格 M 不同）；
   - `set_vertices(pts)` / `vertices()`；`set_faces(f)` / `faces()`；
     `set_colors(c)` / `colors()`；`set_normals(n)` / `normals()`；
   - `has("faces")` 等纳入现有 section 注册表（`codecs/sections.py` 同款 Spec：
     dtype / 压缩 / shuffle / fillvalue）。
3. open 时校验：vertices 必有且轴 0 = M；faces 的 max < M；colors/normals 轴 0 = M。
4. 根 attrs 至少：`MV` = int64 (2,) (M, V)（命名风格随你们：MV / M / MVF 都行，
   请定一个并写进 `docs/schema.md`）；`/meta` 复用现有 `dataset` / `scene` /
   `suffix` / `tags/*`（定位键与 H5Cam 一致，同一场景的相机 h5 和 GT h5 能用
   (dataset, scene) 关联）。
5. 一个从 ply 读入的辅助（可选，放在 SpatialBench 侧也可）：本需求最低要求
   open3d 的 vertices/faces 数组能直接喂给 `set_*`（即 (M,3) f32 / (F,3) i32 /
   (M,3) u8 三种 shape 不拒绝）。

## 消费方用途（评测侧）

- 有 faces → `mesh.sample_points_uniformly(1_000_000)`（open3d 需要 vertices+faces，
  所以 /points/P + /vertices/ptnum 必须能一次性读回完整数组；最大场景实测
  762893 顶点 / 1539281 面 → 11.6 MB，4775139 顶点 / 9812944 面 → 74.5 MB）；
- 无 faces → 顶点直接当点云用（> 1M 时随机降采样）。
- 即：read path 只需要整块 `vertices()` + `faces()` 两个大数组，不需要随机访问。

## 存储期望（供选压缩用）

- 45 文件合计：vertices f32 ≈ 174 MB（double 源按 f32 收敛，误差 1e-7 m，可接受），
  faces i32 ≈ 377 MB（scannetpp 45.3M 面索引，V = 136M），colors ≈ 130 MB，
  normals（7scenes）≈ 38 MB。
- 请给 vertices（gzip4+shuffle？）、faces（raw 还是 gzip？面索引差分后 gzip 收益大，
  但不差分就 raw 也行）、colors（gzip？）、normals 的压缩决策，并用
  `tests/test_compression_policy.py` 同款方式钉住。目标：总盘 < 0.5 GB 为优，
  < 1.8 GB 即可接受（不劣于 ply 也行，转换动机主要是同库统一管理而非省空间）。

## 验证要求

- `tests/` 增加一个测试：合成 vertices/faces/colors/normals → create → close → open →
  逐位相等；点云形态（无 faces）同样 round-trip；越界 face 索引报错；
  缺 vertices 的文件 open 报错。
- SpatialBench 侧转换脚本 + 全量 45 场景验证已完成（2026-11-13，见
  `tests/probes/geo/README.md` §5.4 一节与 `requirements_gt_geometry/REPORT.md`）：
  h5 合计 0.711 GB（< 1.8 GB 门限）、P max diff 2.4e-7 m、Cd/ptnum 逐位精确、
  mesh 采样点数一致 11/11。

## 待拍板（给负责人的问题）

1. **类名与形态**：建议**一个类**（如 `H5Mesh`，faces 可缺省，缺省即点云）——
   消费方本来就按「有面无面」两分支处理，拆两个类（H5Mesh + H5PointCloud）
   会把注册表/section 表/文档翻倍而收益有限；若你们认为点云应独立成类，
   请说明，我们配合。
2. 根 attr 命名（MV / MVF / n_vertices）。
3. create 语义：预声明形状 vs `set_*` 定形。
4. 是否接受 double→float32 收敛（我们接受；若库希望保留 double 请给 dtype 选项）。
5. `is_metric` 要不要带（我们默认米制，若带请允许缺省 True 或强制写）。
