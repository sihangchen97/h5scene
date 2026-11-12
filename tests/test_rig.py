"""Feasibility tests for the H5Cam + H5CamRig merge (prototype in
h5scene/rig.py): N single-camera sequences with DIFFERENT (H, W) -> one
multi-camera H5Scene (T, N, Hmax, Wmax). Verifies padding, mask/depth
completing to common HW, the strict T / section-set checks, and merge speed.
"""
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h5scene.rig import H5Cam, merge_rig  # noqa: E402
from h5scene import H5Scene  # noqa: E402


def _mk_cam(path, T, H, W, seed, has_depth=True):
    """One synthetic single-camera H5Cam: a distinctive per-(t,n) rgb + depth."""
    rng = np.random.default_rng(seed)
    cam = H5Cam(path, T, H, W)
    try:
        b = []
        for t in range(T):
            # rgb frame: a constant block whose value encodes (t, seed) so the
            # merge must place each camera's pixels in the right (t, n) slot
            img = np.full((H, W, 3), (t * 7 + seed) % 250, np.uint8)
            img[0, 0] = (seed, t % 255, 0)  # corner marker
            b.append(cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes())
        cam.set_rgb(b)
        if has_depth:
            # 1.0 + t/100 m -> 1000..1003 mm, comfortably inside uint16 mm range
            d = np.full((T, H, W), 1.0 + t / 100.0, np.float32)
            cam.set_depth(d)
    finally:
        cam.close()


def test_merge_mixed_hw(tmp_path):
    """3 cameras, same T, W identical, DIFFERENT H: merge to (T, N, Hmax, W)
    with top-left (bottom) padding; mask/depth completed so only real pixels
    are valid. The one-side-identical rule is enforced (test_both_axes differ
    rejected below)."""
    T = 4
    sizes = [(64, 96), (48, 96), (32, 96)]   # W=96 shared, H varies -> Hmax=64
    paths = [str(tmp_path / f"c{i}.h5") for i in range(3)]
    for i, (H, W) in enumerate(sizes):
        _mk_cam(paths[i], T, H, W, seed=10 + i)
    out = str(tmp_path / "rig.h5")
    got = merge_rig(paths, out, meta={"dataset": "syn", "scene": "s", "seq": "q"})
    assert got == (T, 3, 64, 96)

    with H5Scene(out) as s:
        assert s.shape == (T, 3, 64, 96)
        H, W = 64, 96
        # camera n occupies its real (h, W) top; the bottom rows are padding
        for n, (h, w) in enumerate(sizes):
            rgb = s.rgb(0, n)
            assert rgb.shape == (H, W, 3)
            assert (rgb[:h, :w] > 0).all()         # real region
            if h != H:
                assert rgb[h:, :].sum() == 0       # padded rows are black
            # depth: real region valid, padded rows 0 (invalid)
            d = s.depth(0, n)
            assert d.shape == (H, W)
            assert (d[:h, :w] > 0).all()           # real mm > 0
            if h != H:
                assert d[h:, :].sum() == 0
            # depth_mask derived: True exactly on the real region
            dm = s.depth_mask(0, n)
            assert dm[:h, :w].all()
            if h != H:
                assert not dm[h:, :].any()
    # both axes differ (H and W all distinct) -> rejected by the strict rule
    a, b = str(tmp_path / "p.h5"), str(tmp_path / "q.h5")
    _mk_cam(a, T, 48, 64, seed=20)
    _mk_cam(b, T, 32, 48, seed=21)
    with pytest.raises(ValueError, match="H or W identical"):
        merge_rig([a, b], str(tmp_path / "both.h5"))
    assert not (tmp_path / "both.h5").exists()


def test_merge_t_mismatch_rejected(tmp_path):
    """Strict check: cameras with different T must be rejected, no file written."""
    _mk_cam(str(tmp_path / "a.h5"), 4, 32, 48, seed=1)
    _mk_cam(str(tmp_path / "b.h5"), 5, 32, 48, seed=2)
    out = str(tmp_path / "rig.h5")
    with pytest.raises(ValueError, match="T mismatch"):
        merge_rig([str(tmp_path / "a.h5"), str(tmp_path / "b.h5")], out)
    assert not Path(out).exists()


def test_merge_section_set_mismatch_rejected(tmp_path):
    """Strict check: a camera missing depth that the others have -> rejected."""
    _mk_cam(str(tmp_path / "a.h5"), 4, 32, 48, seed=1, has_depth=True)
    _mk_cam(str(tmp_path / "b.h5"), 4, 32, 48, seed=2, has_depth=False)
    with pytest.raises(ValueError, match="section sets differ"):
        merge_rig([str(tmp_path / "a.h5"), str(tmp_path / "b.h5")],
                  str(tmp_path / "rig.h5"))


def test_merge_is_metric_mismatch_rejected(tmp_path):
    a, b = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
    _mk_cam(a, 4, 32, 48, seed=1)
    with H5Cam(b, 4, 32, 48, is_metric=False) as cam:
        cam.set_rgb([cv2.imencode(".jpg", np.zeros((32, 48, 3), np.uint8))[1].tobytes()
                     for _ in range(4)])
    with pytest.raises(ValueError, match="is_metric"):
        merge_rig([a, b], str(tmp_path / "rig.h5"))


def test_rig_merge_throughput(tmp_path):
    """Efficiency: timing the N=4, T=20, 640x480 merge (realistic scale)."""
    T, H, W = 20, 480, 640
    paths = []
    for i in range(4):
        p = str(tmp_path / f"c{i}.h5")
        rng = np.random.default_rng(i)
        cam = H5Cam(p, T, H, W)
        try:
            cam.set_rgb([cv2.imencode(
                ".jpg", rng.integers(0, 256, (H, W, 3), np.uint8),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes() for _ in range(T)])
            cam.set_depth(rng.uniform(0.1, 30.0, (T, H, W)).astype(np.float32))
        finally:
            cam.close()
        paths.append(p)
    t0 = time.perf_counter()
    out = str(tmp_path / "big.h5")
    merge_rig(paths, out, meta={"dataset": "syn", "scene": "s"})
    dt = time.perf_counter() - t0
    nframes = T * 4
    print(f"\n[rig throughput] {nframes} frames (4 cams x T{H}x{W}) merged in "
          f"{dt:.2f}s  ({dt / nframes * 1000:.1f} ms/frame, "
          f"{Path(out).stat().st_size / 1e6:.1f} MB out)")
    with H5Scene(out) as s:
        assert s.shape == (T, 4, H, W)
        assert s.rgb(0, 0).shape == (H, W, 3)


if __name__ == "__main__":
    d = Path(tempfile.mkdtemp())
    test_merge_mixed_hw(d)
    test_merge_t_mismatch_rejected(d)
    test_merge_section_set_mismatch_rejected(d)
    test_merge_is_metric_mismatch_rejected(d)
    test_rig_merge_throughput(d)
