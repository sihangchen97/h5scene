"""Tests for codecs/depth.py — the unit-guarded mm-only / meters-only
conversion (the unit is fixed by the entry point, never the dtype), the
sentinel rules (ISSUE #35: one test file per codecs module)."""
import numpy as np
import pytest

from h5scene.codecs import (conf_frame, depth_mask, meters_frame, mm_frame,
                            to_conf, to_meters_mm, to_mm)

MM_MAX = 65535


def test_to_mm_takes_integers_only():
    assert to_mm(np.array([[[500, 1000]]], np.int32))[0, 0].tolist() == [500, 1000]
    assert to_mm(np.array([[[7]]], np.uint16))[0, 0].tolist() == [7]
    with pytest.raises(TypeError, match="set_depth_meters"):
        to_mm(np.array([[[1.0, 2.0]]], np.float32))


def test_to_meters_mm_takes_floats_only():
    mm = to_meters_mm(np.array([[[1.0, 2.0, 0.0]]], np.float32))
    assert mm[0, 0].tolist() == [1000, 2000, 0]
    with pytest.raises(TypeError, match="set_depth_mm"):
        to_meters_mm(np.array([[[500]]], np.int32))


def test_invalid_and_out_of_range():
    # nan/inf/negative -> 0 (invalid)
    mm = to_meters_mm(np.array([[[np.nan, np.inf, -1.0, 70.0, 65.534]]], np.float64))
    assert mm[0, 0].tolist() == [0, 0, 0, 0, 65534]
    # exactly 65535 mm (int or 65.535 m) is the invalid sentinel, not a value
    assert to_mm(np.array([[[65535]]], np.int32))[0, 0].tolist() == [0]
    assert to_meters_mm(np.array([[[65.535]]], np.float64))[0, 0].tolist() == [0]


def test_mm_frame_reads_back_sentinels_as_invalid():
    f = np.array([1000, MM_MAX, 0, 65534], np.uint16)          # (w,) at h=1
    assert mm_frame(f[None], 1, 4)[0].tolist() == [1000, 0, 0, 65534]
    assert np.allclose(meters_frame(f[None], 1, 4)[0], [1.0, 0.0, 0.0, 65.534])


def test_conf_frame_reads_back_01():
    stored = to_conf(np.array([[[0.5, 1.0, 0.0]]], np.float32))[0, 0]
    assert np.allclose(conf_frame(stored[None], 1, 3)[0], [0.5, 1.0, 0.0], atol=2e-3)


def test_to_conf_takes_01_floats_only():
    assert np.allclose(to_conf(np.array([[[0.5]]], np.float32)), 0.5, atol=2e-3)
    # integers are never a 0-1 confidence (the old uint scaling is rejected)
    with pytest.raises(TypeError, match="float 0-1"):
        to_conf(np.array([[[0, 128, 200, 255]]], np.uint8))
    with pytest.raises(ValueError, match="must be 0-1"):
        to_conf(np.array([[[0.5, 1.5]]], np.float32))


def test_depth_mask_excludes_both_sentinels_and_low_conf():
    mm = np.array([1000, MM_MAX, 0, 65534], np.uint16)
    conf = np.array([0.9, 1.0, 0.0, 0.0], np.float32)
    assert depth_mask(mm[None], conf[None], None)[0].tolist() == [True, False, False, False]
    assert depth_mask(mm[None], None, None)[0].tolist() == [True, False, False, True]
    assert depth_mask(mm[None], conf[None], 0.5)[0].tolist() == [True, False, False, False]
    # strict >: conf exactly 0.0 fails even with conf_min=0.0
    assert depth_mask(mm[None], conf[None], 0.0)[0].tolist() == [True, False, False, False]
    conf2 = np.array([0.9, 1.0, 0.5, 0.0], np.float32)
    assert depth_mask(mm[None], conf2[None], 0.5)[0].tolist() == [True, False, False, False]
