from pathlib import Path
"""Tests for H5Cam (single-camera, (T, H, W); the h5cam module — renamed from
test_h5cam.py to mirror the module, ISSUE #35) — round trips, the per-section
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
    with H5Cam.create(out, T, H, W, is_metric=True) as cam:
        assert cam.shape == (T, H, W) and cam.is_metric
        cam.set_rgb(rgb_arr)                    # -> jpg q95
        cam.set_depth_meters(depth_m)
        cam.set_depth_conf_01(conf)
        cam.set_mask(np.ones((T, H, W), bool))
        cam.set_K(np.array([[300., 0., W / 2], [0., 250., H / 2], [0., 0., 1.]], np.float32))
        cam.set_pose_c2w(np.eye(4, dtype=np.float32))
        cam.set_meta(dataset="syn", scene="s", seq="q1", suffix="0",
                     tags={"data_type": "synthetic"})

    with H5Cam.open(out) as cam:
        assert cam.shape == (T, H, W) and cam.is_metric is True
        got = cam.rgb(0)
        assert got.shape == (H, W, 3) and got.dtype == np.uint8
        assert np.abs(got.astype(int) - rgb_arr[0].astype(int)).max() <= 40   # jpg q95 on smooth
        with h5py.File(out) as h:
            b = h["img"]["0"][:]               # stored jpg, not an array
            dec = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            assert isinstance(dec, np.ndarray) and dec.shape == (H, W, 3)
        mm = cam.depth_mm(0)
        assert mm.shape == (H, W) and mm.dtype == np.uint16
        assert mm[0, 0] == 0 and (mm[1] == 1500).all()
        m = cam.depth_meters(0)
        assert m.dtype == np.float32 and np.isclose(m[5, 5], mm[5, 5] / 1000.0)
        # BGR reader: a channel-reversed copy of rgb(t) (exact — both decoded)
        assert (cam.bgr(0) == got[:, :, ::-1]).all()
        c = cam.depth_conf(0)
        assert c.dtype == np.float32 and 0.0 <= c.min() and c.max() <= 1.0
        assert cam.mask(1).all()
        assert cam.K(0).shape == (3, 3) and cam.pose_c2w(1).shape == (4, 4)
        dm = cam.depth_mask(0)
        assert dm.dtype == bool and dm.shape == (H, W) and not dm[0, 0] and dm.any()
        # meta
        assert cam.dataset == "syn" and cam.scene == "s" and cam.seq == "q1"
        assert cam.suffix == "0" and cam.tags == {"data_type": "synthetic"}
        assert cam.src_frame_idxs is None
        assert not cam.has("split")


def test_depth_stored_smaller_aspect(tmp_path):
    T, H, W = 2, 8, 16
    out = str(tmp_path / "cam.h5")
    with H5Cam.create(out, T, H, W) as cam:
        cam.set_depth_mm(np.full((T, 4, 8), 1000.0, np.uint16))     # (h,w)=(4,8), ratio matches
        cam.set_depth_conf_01(np.ones((T, 4, 8), np.float32))
    with h5py.File(out) as h:
        assert h["depth"].shape == (T, 4, 8) and h["depth_conf"].shape == (T, 4, 8)
    with H5Cam.open(out) as cam:
        d = cam.depth_mm(0)
        assert d.shape == (H, W) and (d == 1000).all()          # resized back
        assert cam.depth_mask(0).all()


def test_depth_conf_must_match_depth_hw(tmp_path):
    out = str(tmp_path / "cam.h5")
    with H5Cam.create(out, 2, 8, 16) as cam:
        cam.set_depth_mm(np.full((2, 8, 16), 1000.0, np.uint16))
        with pytest.raises(ValueError, match="!= depth size"):
            cam.set_depth_conf_01(np.ones((2, 4, 8), np.float32))     # (h,w) != depth's (h,w)
    cam = H5Cam.open(out)
    try:
        with pytest.raises(ValueError, match="aspect ratio"):
            cam.set_depth_mm(np.full((2, 3, 16), 1000.0, np.uint16))
    finally:
        cam.close()


def test_missing_and_bad_files(tmp_path):
    p = str(tmp_path / "missing.h5")
    with pytest.raises(FileNotFoundError):
        H5Cam.open(p)
    assert not Path(p).exists()                     # no orphan file
    import h5py as _h
    with _h.File(str(tmp_path / "other.h5"), "w") as h:
        h.create_dataset("x", data=[1])
    with pytest.raises(ValueError, match="not an H5Cam"):
        H5Cam.open(str(tmp_path / "other.h5"))


def test_open_mode_w_is_rejected_not_truncated(tmp_path):
    """The public constructors only mean open ("r"/"r+"): a stray "w" must
    raise BEFORE the file is touched (it would otherwise truncate it)."""
    p = str(tmp_path / "c.h5")
    with H5Cam.create(p, 2, 4, 6) as cam:
        cam.set_depth_meters(np.ones((2, 4, 6), np.float32))
    with pytest.raises(ValueError, match="would truncate"):
        H5Cam(p, mode="w")
    assert H5Cam.open(p).depth_mm(0).shape == (4, 6)    # untouched


def test_del_section(tmp_path):
    """#43: a section can be deleted (rgb: all frames; depth/mask/…: the
    dataset), depth_conf must go first, idempotent, unknown name rejected."""
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 4, 6) as cam:
        cam.set_depth_meters(np.ones((2, 4, 6), np.float32))
        cam.set_depth_conf_01(np.ones((2, 4, 6), np.float32))
        cam.set_mask(np.ones(2, np.bool_))
    with H5Cam.open(str(tmp_path / "c.h5"), "r+") as cam:
        assert cam.has("depth") and cam.has("depth_conf")
        # depth_conf shares depth's (h, w): it must be deleted first
        with pytest.raises(ValueError, match="depth_conf"):
            cam.del_section("depth")
        assert cam.del_section("depth_conf") is cam
        assert cam.del_section("depth") is cam
        assert not cam.has("depth") and not cam.has("depth_conf")
        cam.del_section("mask")
        assert not cam.has("mask")
        assert cam.del_section("mask") is cam          # idempotent
        assert cam.del_section("img") is cam          # absent img: fine
        with pytest.raises(ValueError, match="unknown section"):
            cam.del_section("nope")


def test_is_metric_defaults_false(tmp_path):
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 4, 4) as cam:
        cam.set_rgb_bytes([cv2.imencode(".jpg", np.zeros((4, 4, 3), np.uint8))[1].tobytes()] * 2)
    with H5Cam.open(str(tmp_path / "c.h5")) as cam:
        assert cam.is_metric is False


def test_set_bgr_and_unit_guards(tmp_path):
    """The unit/range guards: a float into set_depth_mm and an integer into
    set_depth_meters / set_depth_conf_01 are rejected (TypeError), and
    set_bgr stores a BGR array as an RGB frame (the red/green swap survives
    a round trip)."""
    T, H, W = 2, 8, 16
    out = str(tmp_path / "cam.h5")
    bgr = np.zeros((T, H, W, 3), np.uint8)
    bgr[:, :, :, 0] = 255                              # BGR channel 0 (blue) full
    with H5Cam.create(out, T, H, W) as cam:
        cam.set_bgr(bgr)
        with pytest.raises(TypeError, match="set_depth_meters"):
            cam.set_depth_mm(np.ones((T, H, W), np.float32))
        with pytest.raises(TypeError, match="set_depth_mm"):
            cam.set_depth_meters(np.ones((T, H, W), np.uint16))
        with pytest.raises(TypeError, match="float 0-1"):
            cam.set_depth_conf_01(np.ones((T, H, W), np.uint8))
    with H5Cam.open(out) as cam:
        got = cam.rgb(0)
        assert got[:, :, 2].min() >= 250 and got[:, :, 0].max() <= 5      # blue -> RGB channel 2 (jpg q95)
        b = cam.bgr(0)
        assert b[:, :, 0].min() >= 250 and b[:, :, 2].max() <= 5          # BGR back (exact vs rgb)
        assert (b == got[:, :, ::-1]).all()
    # set_bgr takes arrays only (bytes / paths are ambiguous colour-order)
    with H5Cam.create(out, 2, 8, 16) as cam:
        with pytest.raises(TypeError, match="uint8 BGR array"):
            cam.set_bgr([b"\xff" * 10] * 2)


def test_meta_unknown_key_rejected(tmp_path):
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 4, 4) as cam:
        from h5scene.h5base import H5Base
        with pytest.raises(ValueError, match="unknown meta key"):
            H5Base.set_meta(cam, "bogus", "x")
        with pytest.raises(ValueError, match="unknown attr"):
            H5Base.set_attr(cam, "bogus", 1)
        assert cam.set_meta(seq="q1") is cam        # the multi-key convenience
        assert cam.seq == "q1"
        H5Base.set_meta(cam, "seq", None)           # base single-key None = remove
        assert cam.seq is None
