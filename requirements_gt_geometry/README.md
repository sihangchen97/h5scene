# requirements_gt_geometry — SpatialBench 的 GT 几何（网格/点云）入 h5scene 需求

- `REQUIREMENTS_GT_GEOMETRY.md` — 需求正文（数据事实、功能需求、验证要求、5 个待拍板问题）
- `sample_plys/` — 每个 source 各 1 个最小样本 ply（可直接 open3d 打开核对 schema）：
  - `scannetpp_mesh_aligned_0.05.ply` — 唯一有面的：vertex f32 + RGB + `element face`（list uchar int）
  - `7scenes_chess_seq-05_gt.ply` — 点云 + **法向**（double ×3）
  - `dtu_rgbd_dtu_scan1_gt.ply` / `nrgbd_kitchen_gt.ply` — 点云（double xyz + uchar RGB）
  - `hiroom_828738.ply` — 点云，M ≠ 1M（1.2M）
- `reference/` — SpatialBench 侧的参照物（非本库代码）：
  - `convert_sparse_to_h5.py` — 现有 H5Cam 转换脚本（API 用法 / 风格参照）
  - `verify_h5_integration.py` — 现有集成验证脚本（验证方式参照）
  - `DATASET.md` — SpatialBench 数据集文档（pointcloud 部分的背景）

消费方：`spatialbench/evaluation/metrics.py` 的 `load_gt_pointcloud_from_mesh`
（open3d 读 ply → 网格采样 1M 点 / 点云直接用 → 与预测 TSDF 点云算 chamfer F-score）。
