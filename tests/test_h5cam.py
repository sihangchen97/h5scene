from pathlib import Path
"""Tests for H5Cam (single-camera, (T, H, W)) — round trips, the per-section
storage policy (stored depth at (h,w), jpg q<=95 rgb), and the meta keys."""

import cv2
import numpy as np
import pytest
import h5py

from h5scene import H5Cam  # noqa: E402
# (no internal imports: tests go through the public API)


def test_round_trip(tmp_path):
    T, H, W = 2, 8, 16
    # a smooth image: jpg q95 lossy but tiny on smooth content (noise would
    # amplify; this pins the encoding path, not a losslessness claim)
    yy, xx = np.mgrid[0:H, 0:W]
    rgb_arr = np.stack([((xx * 16) % 256), ((yy * 32) % 256), np.full((H, W), 128)], -1).astype(np.uint8)
    rgb_arr = np.repeat(rgb_arr[None], T, axis=0)
    depth_m = np.full((T, H, W), 1.5, np.float32)
    depth_m[0, 0, 0] = 70.0                    # > 65.535 m -> must be stored 0
    conf = np.full((T, H, W), 0.5, np.float32)
    out = str(tmp_path / "cam.h5")
    with H5Cam(out, T, H, W, is_metric=True) as cam:
        assert cam.shape == (T, H, W) and cam.is_metric
        cam.set_rgb(rgb_arr)                    # -> jpg q95
        cam.set_depth(depth_m)
        cam.set_depth_conf(conf)
        cam.set_mask(np.ones((T, H, W), bool))
        cam.set_K(np.array([[300., 0., W / 2], [0., 250., H / 2], [0., 0., 1.]], np.float32))
        cam.set_pose(np.eye(4, dtype=np.float32))
        cam.set_meta(dataset="syn", scene="s", seq="q1", suffix="0",
                     tags={"data_type": "synthetic"})

    with H5Cam(out) as cam:
        assert cam.shape == (T, H, W) and cam.is_metric is True
        got = cam.rgb(0)
        assert got.shape == (H, W, 3) and got.dtype == np.uint8
        assert np.abs(got.astype(int) - rgb_arr[0].astype(int)).max() <= 40   # jpg q95 on smooth
        with h5py.File(out) as h:
            b = h["rgb"]["0"][:]               # stored jpg, not an array
            dec = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            assert isinstance(dec, np.ndarray) and dec.shape == (H, W, 3)
        mm = cam.depth(0)
        assert mm.shape == (H, W) and mm.dtype == np.uint16
        assert mm[0, 0] == 0 and (mm[1] == 1500).all()
        m = cam.depth(0, meters=True)
        assert m.dtype == np.float32 and np.isclose(m[5, 5], mm[5, 5] / 1000.0)
        c = cam.depth_conf(0)
        assert c.dtype == np.float32 and 0.0 <= c.min() and c.max() <= 1.0
        assert cam.mask(1).all()
        assert cam.K(0).shape == (3, 3) and cam.pose(1).shape == (4, 4)
        dm = cam.depth_mask(0)
        assert dm.dtype == bool and dm.shape == (H, W) and not dm[0, 0] and dm.any()
        # meta
        assert cam.dataset == "syn" and cam.scene == "s" and cam.seq == "q1"
        assert cam.suffix == "0" and cam.tags == {"data_type": "synthetic"}
        assert cam.frame_indices is None
        assert not cam.has("split")


def test_depth_stored_smaller_aspect(tmp_path):
    T, H, W = 2, 8, 16
    out = str(tmp_path / "cam.h5")
    with H5Cam(out, T, H, W) as cam:
        cam.set_depth(np.full((T, 4, 8), 1000.0, np.uint16))     # (h,w)=(4,8), ratio matches
        cam.set_depth_conf(np.ones((T, 4, 8), np.float32))
    with h5py.File(out) as h:
        assert h["depth"].shape == (T, 4, 8) and h["depth_conf"].shape == (T, 4, 8)
    with H5Cam(out) as cam:
        d = cam.depth(0)
        assert d.shape == (H, W) and (d == 1000).all()          # resized back
        assert cam.depth_mask(0).all()


def test_depth_conf_must_match_depth_hw(tmp_path):
    out = str(tmp_path / "cam.h5")
    with H5Cam(out, 2, 8, 16) as cam:
        cam.set_depth(np.full((2, 8, 16), 1000.0, np.uint16))
        with pytest.raises(ValueError, match="!= depth size"):
            cam.set_depth_conf(np.ones((2, 4, 8), np.float32))     # (h,w) != depth's (h,w)
    cam = H5Cam(out)
    try:
        with pytest.raises(ValueError, match="aspect ratio"):
            cam.set_depth(np.full((2, 3, 16), 1000.0, np.uint16))
    finally:
        cam.close()


def test_missing_and_bad_files(tmp_path):
    p = str(tmp_path / "missing.h5")
    with pytest.raises(FileNotFoundError):
        H5Cam(p)
    assert not Path(p).exists()                     # no orphan file
    import h5py as _h
    with _h.File(str(tmp_path / "other.h5"), "w") as h:
        h.create_dataset("x", data=[1])
    with pytest.raises(ValueError, match="not an H5Cam"):
        H5Cam(str(tmp_path / "other.h5"))


def test_is_metric_defaults_false(tmp_path):
    with H5Cam(str(tmp_path / "c.h5"), 2, 4, 4) as cam:
        cam.set_rgb([cv2.imencode(".jpg", np.zeros((4, 4, 3), np.uint8))[1].tobytes()] * 2)
    with H5Cam(str(tmp_path / "c.h5")) as cam:
        assert cam.is_metric is False


def test_meta_unknown_key_rejected(tmp_path):
    with H5Cam(str(tmp_path / "c.h5"), 2, 4, 4) as cam:
        from h5scene.h5base import H5Base
        with pytest.raises(ValueError, match="unknown meta key"):
            H5Base.set_meta(cam, "bogus", "x")
        with pytest.raises(ValueError, match="unknown attr"):
            H5Base.set_attr(cam, "bogus", 1)
        assert cam.set_meta(seq="q1") is cam        # the multi-key convenience
        assert cam.seq == "q1"
        H5Base.set_meta(cam, "seq", None)           # base single-key None = remove
        assert cam.seq is None
