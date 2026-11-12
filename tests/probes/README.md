# Test assets (probes)

Real scene crops used as test inputs for the h5scene package.

## adt_work_skeleton_seq137/

From `SpatialBench/dataset/sparse_h5/adt/Apartment_release_work_skeleton_seq137_M1292.h5`,
frame 6 (fid 274475577745350) — the I-025 probe scene (near-object "奶粉罐" at
0.07–0.10 m, bottom-right corner):

- `frame6_rgb.jpg` — full 640×640 frame (jpg q95)
- `frame6_depth_mm.npy` — uint16 (640, 640) mm, 0 = invalid, range 59–6208
- `frame6_depth_vis.png` — jet visualization (0 = black)
- `frame6_near_crop.jpg` — the near-crop used in the I-025 investigation

## _i025_probe/

Format-compression test of the near crop (`smoke_image_formats.py` at the
tests/ level, default input) that chose the rgb encoding: **jpg q95**
(2026-11-12): q95 is the size/quality knee — 33 KB/frame vs 112 KB png,
47 dB / 0.995 SSIM; q99/q100 add +45..70% bytes for <0.5 dB (and q100 is
still lossy); q90 shifts 27% of pixels >=1 gray level.

- `frame6_near_crop.jpg` — the source (891x247, 33 KB)
- `frame6_near_crop_fmt_test/` — png / jpg100 / jpg99 / jpg95 / jpg90 variants +
  `frame6_near_crop_compare.png` (ref | png | jpg100 | jpg99 | jpg95 | jpg90, 2x zoom)
