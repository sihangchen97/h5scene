"""Regression tests pinning the reviewed bugs from ISSUE.md. Each test locks
in a behaviour that used to break; rerun with `pixi run test`."""
from pathlib import Path

import os
import cv2
import numpy as np
import pytest
import h5py

from h5scene import H5Base, H5Cam, H5CamRig
from h5scene.codecs import decode_rgb, encode_rgb


def _cam(path, T, H, W, seed, scene="s", depth_hw=None):
    with H5Cam.create(path, T, H, W) as cam:
        b = []
        for t in range(T):
            img = np.full((H, W, 3), (t * 7 + seed) % 250, np.uint8)
            b.append(encode_rgb(img))               # store as proper RGB jpg
        cam.set_rgb(b)
        hh, ww = depth_hw if depth_hw is not None else (H, W)
        d = np.broadcast_to((1.0 + np.arange(T) / 10.0)[:, None, None], (T, hh, ww))
        cam.set_depth_meters(d.astype(np.float32).copy())
        cam.set_meta(dataset="syn", scene=scene, seq=f"q{seed}")
    return path


def test_rig_reads_each_t_not_frame0(tmp_path):
    """#1: every t of a rig must read back its own camera frame t, not frame 0.
    Depth 1,2,3 m per t -> each t reads back its own value (not all frame 0)."""
    T = 3
    # two cameras, W identical, DIFFERENT H -> camera b is padded
    a = _cam(str(tmp_path / "a.h5"), T, 48, 96, seed=1)
    b = str(tmp_path / "b.h5")                      # the smaller camera: distinct depth per frame
    with H5Cam.create(b, T, 32, 96) as c:
        c.set_rgb([encode_rgb(np.full((32, 96, 3), 9, np.uint8))] * T)
        c.set_depth_meters(np.stack([np.full((32, 96), m) for m in (1.0, 2.0, 3.0)]))
        c.set_meta(dataset="syn", scene="s", seq="qb")
    out = str(tmp_path / "rig.h5")
    with H5CamRig.build(out, [a, b]) as rig:
        got = [int(rig.depth_mm(t, 1)[0, 0]) for t in range(T)]
    assert got == [1000, 2000, 3000], f"VDS read wrong frames: {got}"


def test_rig_rejects_downsampled_depth(tmp_path):
    """#2: a camera whose depth is stored at (h, w) < (H, W) cannot be upscaled
    through a VDS block — the build must reject it clearly."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1, depth_hw=(16, 32))
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2)
    with pytest.raises(ValueError, match="upscale|stored at"):
        H5CamRig.build(str(tmp_path / "rig.h5"), [a, b])
    assert not Path(str(tmp_path / "rig.h5")).exists()     # no partial file


def test_rgb_colour_roundtrip_from_file(tmp_path):
    """#8: a red image stored via a file input must read back as RED (R channel
    high), not BGR-swapped. cv2 reads BGR internally; the stored jpg is RGB."""
    red = np.zeros((8, 8, 3), np.uint8)
    red[:, :, 0] = 254                                  # R channel
    p = str(tmp_path / "red.png")
    cv2.imwrite(p, cv2.cvtColor(red, cv2.COLOR_RGB2BGR))   # write a red png
    with H5Cam.create(str(tmp_path / "c.h5"), 1, 8, 8) as cam:
        cam.set_rgb([p])
    with H5Cam.open(str(tmp_path / "c.h5")) as cam:
        got = cam.rgb(0)
    assert int(got[0, 0, 0]) > 200 and int(got[0, 0, 2]) < 60, \
        f"expected red (R high), got BGR-ordered {got[0, 0]}"


def test_failed_build_preserves_output(tmp_path):
    """#5: a failed rig build must not delete or clobber an existing output
    file, and must leave no temp file behind. A scene mismatch fails
    validation BEFORE any write (an output colliding with a camera fails
    even earlier — #36)."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    # a second camera with a DIFFERENT scene -> build must fail
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2, scene="other")
    out = str(tmp_path / "out.h5")
    Path(out).write_bytes(b"keep me")
    with pytest.raises(ValueError, match="scene mismatch"):
        H5CamRig.build(out, [a, b])
    assert Path(out).read_bytes() == b"keep me"   # the existing file survived
    assert not any(p.name.startswith("out.h5.tmp") for p in tmp_path.iterdir())


def test_rgb_frame_count_mismatch(tmp_path):
    """#3: writing fewer frames than T must fail at write time, not at reopen."""
    with H5Cam.create(str(tmp_path / "c.h5"), 3, 8, 8) as cam:
        with pytest.raises(ValueError, match="but T=3"):
            cam.set_rgb([encode_rgb(np.zeros((8, 8, 3), np.uint8))] * 2)


def test_set_meta_type_error_does_not_erase(tmp_path):
    """#4: a type error in set_meta must not have already deleted the value."""
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 8, 8) as cam:
        from h5scene.h5base import H5Base
        H5Base.set_meta(cam, "scene", "s")
        assert cam.scene == "s"
        with pytest.raises(TypeError):
            H5Base.set_meta(cam, "scene", 42)
        assert cam.scene == "s"                 # untouched by the failed write


def test_rig_open_detects_rewritten_camera(tmp_path):
    """#20: opening a rig must verify the camera files still match the stored
    fingerprint (THW + size), not just that they exist."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2)
    out = str(tmp_path / "rig.h5")
    H5CamRig.build(out, [a, b]).close()
    # rewrite camera b with a different T (different content THW)
    b2 = str(tmp_path / "b.h5")
    with H5Cam.create(b2, 5, 32, 64) as c:            # different T -> different content THW
        c.set_rgb([encode_rgb(np.zeros((32, 64, 3), np.uint8))] * 5)
        c.set_meta(dataset="syn", scene="s")
    with pytest.raises((ValueError, FileNotFoundError)):
        H5CamRig.open(out)


def test_src_frame_idxs_length_enforced(tmp_path):
    """#13/#32: src_frame_idxs must be length T (checked by the MetaSpec, not
    silently accepted); a wrong-length array is rejected before any write."""
    with H5Cam.create(str(tmp_path / "c.h5"), 3, 8, 8) as cam:
        with pytest.raises(ValueError, match="length 3"):
            cam.set_meta(src_frame_idxs=np.arange(7))
        cam.set_meta(src_frame_idxs=np.array([5, 9, 12]))
    with H5Cam.open(str(tmp_path / "c.h5")) as cam:
        assert list(cam.src_frame_idxs) == [5, 9, 12]
    # dense (== arange(T)) is dropped on write
    with H5Cam.open(str(tmp_path / "c.h5"), "r+") as cam:
        cam.set_meta(src_frame_idxs=np.arange(3))
        assert cam.src_frame_idxs is None


def test_depth_boundary_65_535m_is_invalid_both_ways(tmp_path):
    """OCR: a depth of exactly 65.535 m (the max representable) must be stored
    as the invalid sentinel (0), not as 65535 which reads back as invalid —
    write and read now agree: 65535 is never a valid depth."""
    from h5scene.codecs import to_meters_mm, to_mm
    assert to_meters_mm(np.array([[65.535]]))[0, 0] == 0       # >= max -> 0 (invalid)
    assert to_meters_mm(np.array([[65.534]]))[0, 0] == 65534  # just under: stored
    assert to_mm(np.array([[65535]], np.int32))[0, 0] == 0  # int mm >= max
    # round trip: the sentinel is read back as invalid (cam H=W=2, T=1)
    with H5Cam.create(str(tmp_path / "c.h5"), 1, 2, 2) as cam:
        cam.set_depth_meters(np.array([[[65.535, 0.0], [0.0, 65.534]]], np.float32))
    with H5Cam.open(str(tmp_path / "c.h5")) as cam:
        d = cam.depth_meters(0)
        assert d[0, 0] == 0.0 and d[1, 1] == 65.534      # 65.535 -> invalid 0
        assert not cam.depth_mask(0)[0, 0] and cam.depth_mask(0)[1, 1]


def test_empty_rgb_sequence_is_a_clear_error(tmp_path):
    """OCR: an empty rgb input must fail with a clear error, not silently
    return (None, None, ()) and crash far downstream."""
    from h5scene.codecs import normalize_rgb
    with pytest.raises(ValueError, match="empty"):
        normalize_rgb([])
    with H5Cam.create(str(tmp_path / "c.h5"), 3, 8, 8) as cam:
        with pytest.raises(ValueError, match="empty"):
            cam.set_rgb([])


def test_set_meta_is_not_overridden(tmp_path):
    """#31: the base set_meta is the ONLY one (no subclass shadow) — both forms
    live on H5Base: single-key (name, value / None) and multi-key (kw=...)."""
    assert H5Cam.set_meta is H5Base.set_meta       # not shadowed anymore
    with H5Cam.create(str(tmp_path / "c.h5"), 3, 8, 8) as cam:
        assert cam.set_meta("scene", "s") is cam
        assert cam.meta("scene") == "s"
        assert cam.set_meta(dataset="d", seq="q") is cam      # multi-key form
        assert cam.meta("dataset") == "d" and cam.meta("seq") == "q"
        assert cam.meta("scene") == "s"                        # untouched keys stay
        cam.set_meta(scene=None)                               # multi-key: None = leave as-is
        assert cam.meta("scene") == "s"
        cam.set_meta("scene", None)                           # single-key remove
        assert cam.meta("scene") is None


def test_rig_rejects_empty_and_duplicate_paths(tmp_path):
    """#18: empty or duplicate cam_paths must fail with a clear error."""
    with pytest.raises(ValueError, match="at least one"):
        H5CamRig.build(str(tmp_path / "r.h5"), [])
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    with pytest.raises(ValueError, match="duplicate"):
        H5CamRig.build(str(tmp_path / "r.h5"), [a, a])
    with pytest.raises(ValueError, match="at least one"):
        H5CamRig.build(str(tmp_path / "r.h5"), [])


def test_rig_output_path_must_not_be_a_camera(tmp_path):
    """#36: a SUCCESSFUL build whose output path is one of its own cameras
    used to overwrite that camera (its data lost, VDS pointing at itself)."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2)
    size_before = Path(a).stat().st_size
    with pytest.raises(ValueError, match="one of its own cameras"):
        H5CamRig.build(a, [a, b])
    assert Path(a).stat().st_size == size_before     # the camera survived
    with H5Cam.open(a) as c:
        assert c.shape == (3, 32, 64)
    assert not any(p.name.startswith("a.h5.tmp") for p in tmp_path.iterdir())


def test_rig_build_creates_output_dir(tmp_path):
    """#42: a build into a not-yet-existing output directory must succeed,
    like H5Cam.create."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2)
    out = str(tmp_path / "deep" / "nested" / "rig.h5")
    H5CamRig.build(out, [a, b]).close()
    assert Path(out).exists()
    with H5CamRig.open(out) as rig:
        assert rig.shape == (3, 2, 32, 64)


def test_failed_build_leaves_no_empty_output_dir(tmp_path):
    """A build that fails validation must not leave an empty output directory
    behind (the dir is created by the base create, only when a file is
    actually being written)."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2, scene="other")
    out = str(tmp_path / "deep" / "nested" / "rig.h5")
    assert not (tmp_path / "deep").exists()
    with pytest.raises(ValueError, match="scene mismatch"):
        H5CamRig.build(out, [a, b])
    assert not (tmp_path / "deep").exists()          # no empty dir left


def test_set_meta_multi_key_validates_all_before_writing(tmp_path):
    """#37: a type error on the SECOND key of a multi-key set_meta must not
    have written the first one (validate every key before any write)."""
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 8, 8) as cam:
        with pytest.raises(TypeError):
            cam.set_meta(split="train", scene=42)
        assert cam.meta("split") is None            # nothing was written
        cam.set_meta(split="train")
        with pytest.raises(TypeError):
            cam.set_meta(split="train", scene=42)
        assert cam.meta("split") == "train"         # the first key still fine
        with pytest.raises(ValueError, match="unknown meta key"):
            cam.set_meta(dataset="d", bogus="x")
        assert cam.meta("dataset") is None


def test_set_meta_mixing_forms_is_an_error(tmp_path):
    """#38: set_meta(name, value, kw=...) used to silently drop the keywords;
    it must raise (one form or the other)."""
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 8, 8) as cam:
        with pytest.raises(ValueError, match="one form"):
            cam.set_meta("split", "x", scene="y")
        assert cam.meta("split") is None            # nothing written either way
        assert cam.meta("scene") is None


def test_conf_rejects_integers_and_out_of_range(tmp_path):
    """#41 closed: an integer confidence is never a 0-1 value — it is
    REJECTED (TypeError), not silently scaled/clipped; the writer takes
    0-1 floats only (set_depth_conf_01)."""
    from h5scene.codecs import to_conf
    with pytest.raises(TypeError, match="float 0-1"):
        to_conf(np.array([[0, 128, 200, 255]], np.uint8))
    with pytest.raises(ValueError, match="must be 0-1"):
        to_conf(np.array([[0.5, 1.5]], np.float32))
    # round trip through the file
    with H5Cam.create(str(tmp_path / "c.h5"), 1, 1, 4) as cam:
        cam.set_depth_meters(np.array([[[1.0, 1.0, 1.0, 1.0]]], np.float32))
        cam.set_depth_conf_01(np.array([[0.0, 0.5, 1.0, 1.0]]))
        with pytest.raises(TypeError, match="float 0-1"):
            cam.set_depth_conf_01(np.array([[0, 128, 200, 255]], np.uint8))
    with H5Cam.open(str(tmp_path / "c.h5")) as cam:
        c = cam.depth_conf(0)
        assert abs(float(c[0, 1]) - 0.5) < 1e-3 and float(c[0, 2]) == 1.0
        assert not cam.depth_mask(0)[0, 0] and cam.depth_mask(0)[0, 1]


def test_rig_survives_cp_and_moves_with_cameras(tmp_path):
    """#40: a rig copied together with its cameras (byte-identical, new
    mtime) must still open; a camera rewritten to a different THW must be
    rejected. cam_files are stored relative to the rig's dir."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2)
    out = str(tmp_path / "rig.h5")
    H5CamRig.build(out, [a, b]).close()
    with h5py.File(out) as h:
        assert [str(p) for p in h.attrs["cam_files"]] == ["a.h5", "b.h5"]
        assert h.attrs["cam_fp"].shape == (2, 3)    # (T, H, W) content
    # byte-identical copy (cp) must open fine — the fingerprint is content-based
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    Path(out).replace(moved / "rig.h5")              # the rig moves too
    os.system(f"cp {a} {moved}/a.h5 && cp {b} {moved}/b.h5")
    os.utime(moved / "a.h5")                         # force a different mtime (content unchanged — must still open)
    with H5CamRig.open(str(moved / "rig.h5")) as rig:
        assert rig.shape == (3, 2, 32, 64)
        assert int(rig.depth_mm(2, 0)[0, 0]) == 1200   # VDS reads the real data
    # a rewritten camera (different T -> different THW content) must be caught
    with H5Cam.create(str(moved / "b.h5"), 5, 32, 64) as c:
        c.set_rgb([encode_rgb(np.zeros((32, 64, 3), np.uint8))] * 5)
        c.set_meta(dataset="syn", scene="s")
    with pytest.raises((ValueError, FileNotFoundError)):
        H5CamRig.open(str(moved / "rig.h5"))
    # a missing camera still raises clearly
    (moved / "a.h5").unlink()
    with pytest.raises(FileNotFoundError):
        H5CamRig.open(str(moved / "rig.h5"))


def test_depth_zero_hw_is_rejected_cleanly(tmp_path):
    """#74: (T, 0, 0) depth must raise a clear ValueError (not h5py's
    'All chunk dimensions must be positive')."""
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 4, 6) as cam:
        with pytest.raises(ValueError, match="aspect ratio"):
            cam.set_depth_mm(np.zeros((2, 0, 0), np.int32))
        with pytest.raises(ValueError, match="aspect ratio"):
            cam.set_depth_conf_01(np.zeros((2, 0, 0), np.float32))


def test_create_rejects_non_integral_thw(tmp_path):
    """#75: a float T/H/W is rejected (int() would silently truncate 2.7 -> 2)."""
    p = str(tmp_path / "c.h5")
    for bad in ((2.7, 8, 8), (2, 0.5, 8), (2, 8, "8")):
        with pytest.raises(ValueError, match="integral"):
            H5Cam.create(p, *bad)
    assert not Path(p).exists()                  # no half-made file
    # numpy ints and exact floats are fine
    with H5Cam.create(p, np.int64(2), 8, 8) as cam:
        assert cam.T == 2


def test_open_requires_is_metric(tmp_path):
    """#76: a file with only attrs.THW is NOT an H5Cam (is_metric is
    always written by create, so its absence means a foreign file)."""
    import h5py as _h
    p = str(tmp_path / "foreign.h5")
    with _h.File(p, "w") as h:
        h.attrs["THW"] = np.array((2, 8, 8), np.int64)
        h.create_group("meta")
    with pytest.raises(ValueError, match="not an H5Cam file"):
        H5Cam.open(p)
    with _h.File(p, "w") as h:                  # same for a rig-shaped file
        h.attrs.update(TNHW=np.array((2, 1, 8, 8), np.int64),
                       cam_files=np.array(["a.h5"], dtype=_h.string_dtype()),
                       cam_hw=np.array([(8, 8)], np.int64),
                       cam_fp=np.array([(2, 8, 8)], np.int64))
        h.create_group("meta")
    with pytest.raises(ValueError, match="not an H5CamRig file"):
        H5CamRig.open(p)


def test_camera_meta_edit_does_not_break_rig_open(tmp_path):
    """#71: the fingerprint is the camera THW, not the file size — a
    metadata edit (which grows the HDF5 file) must NOT break the rig."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2)
    out = str(tmp_path / "rig.h5")
    H5CamRig.build(out, [a, b]).close()
    before = os.path.getsize(a)
    with H5Cam.open(a, "r+") as c:
        c.set_meta("split", "train")            # grows the file
    assert os.path.getsize(a) > before
    with H5CamRig.open(out) as rig:             # must still open
        assert rig.shape == (3, 2, 32, 64)
        assert int(rig.depth_mm(2, 0)[0, 0]) == 1200
