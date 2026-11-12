"""File size + detail loss for one image saved as PNG / JPG q100 / q99 / q95 / q90.

Verifies the h5scene rgb encoding choice: jpg q95 (h5scene/__init__.py).
The measured winner on tests/probes/_i025_probe/frame6_near_crop.jpg (2026-11-12):
  fmt     size_KB   psnr      ssim    max   mean  frac>1  frac>5
  png     112.1     inf       1.0     0     0.0   0%      0%      (lossless, 3.4x bigger)
  q95     32.8      46.98     0.9951  40    0.23  8.9%    1.9%    <- chosen: knee of the curve
  q99     47.8      47.34     0.9963  38    0.23  9.6%    1.7%    (+45% size for +0.36 dB, not worth it)
  q100    55.9      47.43     0.9965  37    0.22  8.7%    1.7%    (+70% size for +0.45 dB; NOT lossless)
  q90     24.7      43.39     0.9852  55    0.66  27.1%   3.5%    (27% of pixels shifted >=1 gray level)
q95 -> q99 -> q100 gains are within measurement noise (mean error 0.229 -> 0.216)
while cost +45..70% in bytes; q90 shifts a quarter of the pixels.

Usage:
    pixi run python tests/smoke_image_formats.py [source_image]

Prints mean/RMSE/psnr/ssim/percentile/max/fraction-over-threshold/per-channel
metrics per format vs the source, pair-wise numbers, and writes a 6-panel
zoomed crop for visual inspection.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = np.abs(a - b)
    per = diff.max(axis=2)
    p = psnr(a, b, data_range=255)
    return {
        "psnr": "inf" if np.isinf(p) else round(float(p), 2),
        "ssim": round(float(ssim(a, b, channel_axis=2, data_range=255)), 5),
        "mean": round(float(diff.mean()), 3),
        "rmse": round(float(np.sqrt(((a - b) ** 2).mean())), 3),
        "p95": round(float(np.percentile(per, 95)), 1),
        "p99": round(float(np.percentile(per, 99)), 1),
        "max": int(diff.max()),
        "frac>1": round(100.0 * (per > 1).mean(), 2),
        "frac>5": round(100.0 * (per > 5).mean(), 2),
        "frac>10": round(100.0 * (per > 10).mean(), 2),
        "R": round(float(diff[..., 0].mean()), 3),
        "G": round(float(diff[..., 1].mean()), 3),
        "B": round(float(diff[..., 2].mean()), 3),
    }


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "tests/probes/_i025_probe/frame6_near_crop.jpg")
    src = src.resolve()
    arr = np.asarray(Image.open(src).convert("RGB"))
    h, w, _ = arr.shape

    out_dir = src.parent / f"{src.stem}_fmt_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)

    variants: dict[str, np.ndarray] = {}
    sizes: dict[str, int] = {}
    for name, ext, save in [
        ("png", "png", lambda im, p: im.save(p)),
        ("jpg100", "jpg", lambda im, p: im.save(p, quality=100)),
        ("jpg99", "jpg", lambda im, p: im.save(p, quality=99)),
        ("jpg95", "jpg", lambda im, p: im.save(p, quality=95)),
        ("jpg90", "jpg", lambda im, p: im.save(p, quality=90)),
    ]:
        p = out_dir / f"{src.stem}_{name}.{ext}"
        save(img, p)
        sizes[name] = p.stat().st_size
        variants[name] = np.asarray(Image.open(p).convert("RGB"))

    NAMES = ("png", "jpg100", "jpg99", "jpg95", "jpg90")
    print(f"source: {src}  {w}x{h}")
    hdr = (f"{'fmt':<9}{'size_KB':>9}   {'mean':>6} {'rmse':>6} {'psnr':>6} {'ssim':>7} "
           f"{'p95':>4} {'p99':>4} {'max':>4} | {'>1':>6} {'>5':>6} {'>10':>6} | {'R/G/B':>12}")
    print(hdr)
    for name in NAMES:
        m = metrics(arr, variants[name])
        print(f"{name:<9}{sizes[name] / 1024:>9.1f}   {m['mean']:>6.3f} {m['rmse']:>6.3f} "
              f"{m['psnr']:>6} {m['ssim']:>7.5f} {m['p95']:>4.1f} {m['p99']:>4.1f} {m['max']:>4} "
              f"| {m['frac>1']:>5.2f}% {m['frac>5']:>5.2f}% {m['frac>10']:>5.2f}% "
              f"| {m['R']:.3f}/{m['G']:.3f}/{m['B']:.3f}")

    print("pair-wise (psnr / ssim):")
    for a, b in [("png", "jpg95"), ("jpg95", "jpg99"), ("jpg99", "jpg100"), ("jpg95", "jpg90")]:
        m = metrics(variants[a], variants[b])
        print(f"  {a} vs {b}: psnr={m['psnr']}  ssim={m['ssim']}  "
              f"max={m['max']}  mean={m['mean']}")

    # 6-panel comparison (ref + 5 variants): crop the most-detailed square
    # region, 2x zoom. Falls back to the whole image for small inputs.
    size = min(h, w)
    if size >= 240:
        side = 200
        var = np.var(arr.astype(np.float32), axis=2)
        step = 20
        best, best_v = (0, 0), -1.0
        for y in range(0, h - side + 1, step):
            for x in range(0, w - side + 1, step):
                v = float(var[y:y + side, x:x + side].mean())
                if v > best_v:
                    best, best_v = (y, x), v
        y0, x0 = best
    else:
        side = size
        x0 = (w - size) // 2
        y0 = 0
    crops = [arr[y0:y0 + side, x0:x0 + side]]
    for name in NAMES:
        crops.append(variants[name][y0:y0 + side, x0:x0 + side])
    panels = [Image.fromarray(c).resize((400, 400), Image.NEAREST) for c in crops]
    canvas = Image.new("RGB", (6 * 405 - 5, 400), "black")
    for i, p in enumerate(panels):
        canvas.paste(p, (i * 405, 0))
    cmp_path = out_dir / f"{src.stem}_compare.png"
    canvas.save(cmp_path)
    print(f"\ncomparison (ref | png | jpg100 | jpg99 | jpg95 | jpg90, 2x zoom, "
          f"{side}x{side} @ {x0},{y0}): {cmp_path}")


if __name__ == "__main__":
    main()
