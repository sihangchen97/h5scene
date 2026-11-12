"""Tests for H5CamRig (N x H5Cam -> (T, N, Hmax, Wmax)) in h5scene/h5camrig.py.
H5CamRig is a references-only file (0 bytes of data on disk): the fixed-shape
sections (/depth /mask /K /T /depth_conf) are HDF5 VIRTUAL DATASETS
referencing the camera files (compressed source mapped directly, pad via
fillvalue) and /img/{t}_{n} are ExternalLinks to each camera's /img/{t}
(jpg bytes, never re-encoded; rgb(t, n) decodes at the camera's native size).

Covers: mixed-H merge (one axis identical, other = max + pad), the strict
checks (T / is_metric / section-set / both-axes-differ), VDS zero-copy +
fillvalue pad, the source-file guard, and merge throughput.
"""
from pathlib import Path
import os
import tempfile
import time

import cv2
import h5py
import numpy as np
import pytest

from h5scene import H5Cam, H5CamRig  # noqa: E402


def _mk_cam(path, T, H, W, seed, has_depth=True):
    """One synthetic single-camera H5Cam: a constant rgb (value encodes seed)
    + an all-valid depth (1.0 + t/100 m -> comfortably inside uint16 mm)."""
    cam = H5Cam(path, T, H, W)
    try:
        b = []
        for t in range(T):
            img = np.full((H, W, 3), (t * 7 + seed) % 250, np.uint8)
            img[0, 0] = (seed, t % 255, 0)          # corner marker
            b.append(cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes())
        cam.set_rgb(b)
        if has_depth:
            cam.set_depth_meters(np.full((T, H, W), 1.0 + t / 100.0, np.float32))
        cam.set_meta(dataset="syn", scene="s", seq=f"q{seed}")   # seqs differ -> the rig merges them
    finally:
        cam.close()


def test_rig_mixed_hw(tmp_path):
    """3 cameras, same T, W identical, DIFFERENT H -> (T, 3, Hmax, W).
    /img is an ExternalLink -> each cam read at its NATIVE (h, w) (nothing
    copied/stored, no re-encode); /depth /mask are VDS -> padded to (Hmax, W)
    top-left via fillvalue, real region exact."""
    T = 4
    sizes = [(64, 96), (48, 96), (32, 96)]          # W=96 shared, H varies -> Hmax=64
    paths = [str(tmp_path / f"c{i}.h5") for i in range(3)]
    for i, (H, W) in enumerate(sizes):
        _mk_cam(paths[i], T, H, W, seed=10 + i)
    out = str(tmp_path / "rig.h5")
    rig = H5CamRig(out, paths, suffix="q")
    try:
        assert rig.shape == (T, 3, 64, 96)
        H, W = 64, 96
        for n, (h, w) in enumerate(sizes):
            # rgb: ExternalLink -> the camera's native (h, w), not padded
            rgb = rig.rgb(0, n)
            assert rgb.shape == (h, w, 3)
            assert (rgb > 0).all()                   # whole real region
            # depth / mask: VDS -> padded to (Hmax, W), top H rows real
            d = rig.depth_mm(0, n)
            assert d.shape == (H, W)
            assert (d[:h, :w] > 0).all()             # real mm > 0
            if h != H:
                assert d[h:, :].sum() == 0           # padded rows invalid
            dm = rig.depth_mask(0, n)
            assert dm[:h, :w].all()
            if h != H:
                assert not dm[h:, :].any()
        # /depth is a VDS (0 bytes); /img entries are ExternalLinks (no copy)
        with h5py.File(out, "r") as h:
            assert h["depth"].id.get_storage_size() == 0
            for k in h["img"]:
                assert isinstance(h.get(f"img/{k}", getlink=True), h5py.ExternalLink)
            assert "cam_files" in h.attrs and "cam_hw" in h.attrs
            assert [tuple(x) for x in h.attrs["cam_hw"]] == sizes
    finally:
        rig.close()


def test_rig_t_mismatch_rejected(tmp_path):
    _mk_cam(str(tmp_path / "a.h5"), 4, 32, 48, seed=1)
    _mk_cam(str(tmp_path / "b.h5"), 5, 32, 48, seed=2)
    with pytest.raises(ValueError, match="T mismatch"):
        H5CamRig(str(tmp_path / "rig.h5"), [str(tmp_path / "a.h5"), str(tmp_path / "b.h5")])


def test_rig_section_set_mismatch_rejected(tmp_path):
    _mk_cam(str(tmp_path / "a.h5"), 4, 32, 48, seed=1, has_depth=True)
    _mk_cam(str(tmp_path / "b.h5"), 4, 32, 48, seed=2, has_depth=False)
    with pytest.raises(ValueError, match="section sets differ"):
        H5CamRig(str(tmp_path / "rig.h5"), [str(tmp_path / "a.h5"), str(tmp_path / "b.h5")])


def test_rig_is_metric_mismatch_rejected(tmp_path):
    a = str(tmp_path / "a.h5")
    _mk_cam(a, 4, 32, 48, seed=1)
    b = str(tmp_path / "b.h5")
    with H5Cam(b, 4, 32, 48, is_metric=True) as cam:
        cam.set_rgb([cv2.imencode(".jpg", np.zeros((32, 48, 3), np.uint8))[1].tobytes()
                     for _ in range(4)])
    with pytest.raises(ValueError, match="is_metric"):
        H5CamRig(str(tmp_path / "rig.h5"), [a, b])


def test_rig_both_axes_differ_rejected(tmp_path):
    """H AND W both differ across cameras -> rejected (one axis must match)."""
    a, b = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
    _mk_cam(a, 4, 48, 64, seed=1)
    _mk_cam(b, 4, 32, 48, seed=2)
    out = str(tmp_path / "rig.h5")
    with pytest.raises(ValueError, match="H or W identical"):
        H5CamRig(out, [a, b])
    assert not os.path.exists(out)


def test_rig_source_file_guard(tmp_path):
    """VDS would silently read fillvalue if a source vanishes: H5CamRig(open)
    must reject a missing cam_files entry instead."""
    a = str(tmp_path / "a.h5")
    _mk_cam(a, 4, 32, 48, seed=1)
    out = str(tmp_path / "rig.h5")
    rig = H5CamRig(out, [a])
    rig.close()
    os.rename(a, a + ".moved")
    with pytest.raises(FileNotFoundError, match="missing camera source"):
        H5CamRig(out)

def test_rig_dataset_scene_mismatch_rejected(tmp_path):
    a = str(tmp_path / "a.h5")
    _mk_cam(a, 4, 32, 48, seed=1)
    b = str(tmp_path / "b.h5")                      # different scene name
    with H5Cam(b, 4, 32, 48) as cam:
        cam.set_rgb([cv2.imencode(".jpg", np.zeros((32, 48, 3), np.uint8))[1].tobytes()
                     for _ in range(4)])
        cam.set_meta(dataset="syn", scene="other")
    with pytest.raises(ValueError, match="scene mismatch"):
        H5CamRig(str(tmp_path / "rig.h5"), [a, b])


def test_rig_throughput(tmp_path):
    """Merge throughput at realistic scale (4 cams, T=20, 480x640, rgb+depth).
    The depth section is a VDS -> 0 bytes; the cost is the /img physical copy
    (verbatim for the full-size cam, decode+pad+re-encode for the two smaller)."""
    T, W = 20, 640
    sizes = [(480, W), (480, W), (360, W), (240, W)]
    paths = []
    for i, (H, W_) in enumerate(sizes):
        p = str(tmp_path / f"c{i}.h5")
        cam = H5Cam(p, T, H, W_)
        rng = np.random.default_rng(i)
        try:
            cam.set_rgb([cv2.imencode(".jpg", rng.integers(0, 256, (H, W_, 3), np.uint8),
                                      [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
                         for _ in range(T)])
            cam.set_depth_meters(rng.uniform(0.1, 30.0, (T, H, W_)).astype(np.float32))
        finally:
            cam.close()
        paths.append(p)
    out = str(tmp_path / "big.h5")
    t0 = time.perf_counter()
    rig = H5CamRig(out, paths)
    dt = time.perf_counter() - t0
    try:
        nframes = T * 4
        print(f"\n[rig throughput] {nframes} frames (4 cams x T{W}x{480}) in "
              f"{dt:.2f}s ({dt / nframes * 1000:.1f} ms/frame), "
              f"out {Path(out).stat().st_size / 1e6:.1f} MB")
        assert rig.shape == (T, 4, 480, W)
        assert rig.rgb(0, 0).shape == (480, W, 3)
        with h5py.File(out, "r") as h:
            assert h["depth"].id.get_storage_size() == 0      # VDS, not copied
    finally:
        rig.close()


if __name__ == "__main__":
    import pytest as _p
    _p.main([__file__, "-q"])
