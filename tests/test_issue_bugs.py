"""Regression tests for the reviewed bugs (ISSUE.md, 2026-11-13). Each test pins
a previously-broken behaviour; line refs in ISSUE.md refer to the old layout.
Rerun: pixi run -e test test.
"""
from pathlib import Path

import os
import cv2
import numpy as np
import pytest
import h5py

from h5scene import H5Cam, H5CamRig
from h5scene.codecs import decode_rgb, encode_rgb


def _cam(path, T, H, W, seed, scene="s", depth_hw=None):
    with H5Cam(path, T, H, W) as cam:
        b = []
        for t in range(T):
            img = np.full((H, W, 3), (t * 7 + seed) % 250, np.uint8)
            b.append(encode_rgb(img))               # store as proper RGB jpg
        cam.set_rgb(b)
        hh, ww = depth_hw if depth_hw is not None else (H, W)
        d = np.broadcast_to((1.0 + np.arange(T) / 10.0)[:, None, None], (T, hh, ww))
        cam.set_depth(d.astype(np.float32).copy())
        cam.set_meta(dataset="syn", scene=scene, seq=f"q{seed}")
    return path


def test_rig_per_frame_not_frame0(tmp_path):
    """#1: the per-frame VDS block must select camera frame t, not frame 0.
    Depth 1,2,3 m per t -> each t reads back its own value (not all frame 0)."""
    T = 3
    # two cameras, W identical, DIFFERENT H -> the per-frame branch runs
    a = _cam(str(tmp_path / "a.h5"), T, 48, 96, seed=1)
    b = str(tmp_path / "b.h5")                      # the smaller camera: distinct depth per frame
    with H5Cam(b, T, 32, 96) as c:
        c.set_rgb([encode_rgb(np.full((32, 96, 3), 9, np.uint8))] * T)
        c.set_depth(np.stack([np.full((32, 96), m) for m in (1.0, 2.0, 3.0)]))
        c.set_meta(dataset="syn", scene="s", seq="qb")
    out = str(tmp_path / "rig.h5")
    with H5CamRig(out, [a, b]) as rig:
        got = [int(rig.depth(t, 1)[0, 0]) for t in range(T)]
    assert got == [1000, 2000, 3000], f"per-frame VDS read wrong frames: {got}"


def test_rig_rejects_downsampled_depth(tmp_path):
    """#2: a camera whose depth is stored at (h, w) < (H, W) cannot be upscaled
    through a VDS block — the build must reject it clearly."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1, depth_hw=(16, 32))
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2)
    with pytest.raises(ValueError, match="upscale|stored at"):
        H5CamRig(str(tmp_path / "rig.h5"), [a, b])
    assert not Path(str(tmp_path / "rig.h5")).exists()     # no partial file


def test_rgb_colour_roundtrip_from_file(tmp_path):
    """#8: a red image stored via a file input must read back as RED (R channel
    high), not BGR-swapped. cv2 reads BGR internally; the stored jpg is RGB."""
    red = np.zeros((8, 8, 3), np.uint8)
    red[:, :, 0] = 254                                  # R channel
    p = str(tmp_path / "red.png")
    cv2.imwrite(p, cv2.cvtColor(red, cv2.COLOR_RGB2BGR))   # write a red png
    with H5Cam(str(tmp_path / "c.h5"), 1, 8, 8) as cam:
        cam.set_rgb([p])
    with H5Cam(str(tmp_path / "c.h5")) as cam:
        got = cam.rgb(0)
    assert int(got[0, 0, 0]) > 200 and int(got[0, 0, 2]) < 60, \
        f"expected red (R high), got BGR-ordered {got[0, 0]}"


def test_failed_build_preserves_output(tmp_path):
    """#5: a failed rig build must not delete an existing output file, and must
    leave no temp file behind. Put the output AT a camera path to prove the
    old os.remove-before-validate bug is gone."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    # a second camera with a DIFFERENT scene -> build must fail
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2, scene="other")
    out = str(tmp_path / "a.h5")            # same as camera a's path
    with pytest.raises(ValueError, match="scene mismatch"):
        H5CamRig(out, [a, b])
    assert Path(out).exists()               # the camera file survived
    with H5Cam(out) as c:                   # and is still a valid camera
        assert c.shape == (3, 32, 64)
    assert not any(p.name.startswith("a.h5.tmp") for p in tmp_path.iterdir())


def test_rgb_frame_count_mismatch(tmp_path):
    """#3: writing fewer frames than T must fail at write time, not at reopen."""
    with H5Cam(str(tmp_path / "c.h5"), 3, 8, 8) as cam:
        with pytest.raises(ValueError, match="but T=3"):
            cam.set_rgb([encode_rgb(np.zeros((8, 8, 3), np.uint8))] * 2)


def test_set_meta_type_error_does_not_erase(tmp_path):
    """#4: a type error in set_meta must not have already deleted the value."""
    with H5Cam(str(tmp_path / "c.h5"), 2, 8, 8) as cam:
        from h5scene.h5base import H5Base
        H5Base.set_meta(cam, "scene", "s")
        assert cam.scene == "s"
        with pytest.raises(TypeError):
            H5Base.set_meta(cam, "scene", 42)
        assert cam.scene == "s"                 # untouched by the failed write


def test_rig_open_detects_rewritten_camera(tmp_path):
    """#20: opening a rig must verify the camera files still match the stored
    fingerprint (size/mtime), not just that they exist."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2)
    out = str(tmp_path / "rig.h5")
    H5CamRig(out, [a, b]).close()
    # rewrite camera b with a different T (new size + mtime)
    b2 = str(tmp_path / "b.h5")
    with H5Cam(b2, 5, 32, 64) as c:            # different T -> different size + mtime
        c.set_rgb([encode_rgb(np.zeros((32, 64, 3), np.uint8))] * 5)
        c.set_meta(dataset="syn", scene="s")
    with pytest.raises((ValueError, FileNotFoundError)):
        H5CamRig(out)


def test_frame_indices_length_enforced(tmp_path):
    """#13/#32: frame_indices must be length T (checked by the MetaSpec, not
    silently accepted); a wrong-length array is rejected before any write."""
    with H5Cam(str(tmp_path / "c.h5"), 3, 8, 8) as cam:
        with pytest.raises(ValueError, match="length 3"):
            cam.set_meta(frame_indices=np.arange(7))
        cam.set_meta(frame_indices=np.array([5, 9, 12]))
    with H5Cam(str(tmp_path / "c.h5")) as cam:
        assert list(cam.frame_indices) == [5, 9, 12]
    # dense (== arange(T)) is dropped on write
    with H5Cam.open(str(tmp_path / "c.h5"), "r+") as cam:
        cam.set_meta(frame_indices=np.arange(3))
        assert cam.frame_indices is None


def test_rig_rejects_empty_and_duplicate_paths(tmp_path):
    """#18: empty or duplicate cam_paths must fail with a clear error."""
    with pytest.raises(ValueError, match="at least one"):
        H5CamRig(str(tmp_path / "r.h5"), [])
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    with pytest.raises(ValueError, match="duplicate"):
        H5CamRig(str(tmp_path / "r.h5"), [a, a])
    with pytest.raises(ValueError, match="at least one"):
        H5CamRig(str(tmp_path / "r.h5"), [])
