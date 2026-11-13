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
        cam.set_rgb_bytes(b)
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
        c.set_rgb_bytes([encode_rgb(np.full((32, 96, 3), 9, np.uint8))] * T)
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
        cam.set_rgb_paths([p])
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
        with pytest.raises(ValueError, match="T=3"):
            cam.set_rgb_bytes([encode_rgb(np.zeros((8, 8, 3), np.uint8))] * 2)


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
    """#20: opening a rig must verify each camera's THW is still
    (T, *cam_hw[n]) and its mapped sections intact, not just that the file
    exists."""
    a = _cam(str(tmp_path / "a.h5"), 3, 32, 64, seed=1)
    b = _cam(str(tmp_path / "b.h5"), 3, 32, 64, seed=2)
    out = str(tmp_path / "rig.h5")
    H5CamRig.build(out, [a, b]).close()
    # rewrite camera b with a different T (different content THW)
    b2 = str(tmp_path / "b.h5")
    with H5Cam.create(b2, 5, 32, 64) as c:            # different T -> different content THW
        c.set_rgb_bytes([encode_rgb(np.zeros((32, 64, 3), np.uint8))] * 5)
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
    from h5scene.codecs import normalize_img_bytes
    with pytest.raises(ValueError, match="empty"):
        normalize_img_bytes([])
    with H5Cam.create(str(tmp_path / "c.h5"), 3, 8, 8) as cam:
        with pytest.raises(ValueError, match="empty"):
            cam.set_rgb_bytes([])


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
        assert h.attrs["cam_hw"].tolist() == [[32, 64], [32, 64]]    # per-camera (H, W)
    # byte-identical copy (cp) must open fine — open checks the THW, not the file size
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
        c.set_rgb_bytes([encode_rgb(np.zeros((32, 64, 3), np.uint8))] * 5)
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
    """#75/#80: a non-integral T/H/W is a TypeError (int() would silently
    truncate 2.7 -> 2); bool is rejected too (operator.index(True) == 1)."""
    p = str(tmp_path / "c.h5")
    for bad in ((2.7, 8, 8), (2, 0.5, 8), (2, 8, "8"), (True, 8, 8), (2, False, 8)):
        with pytest.raises(TypeError, match="integral"):
            H5Cam.create(p, *bad)
    assert not Path(p).exists()                  # no half-made file
    # numpy ints are fine
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
                       cam_hw=np.array([(8, 8)], np.int64))
        h.create_group("meta")
    with pytest.raises(ValueError, match="not an H5CamRig file"):
        H5CamRig.open(p)


def test_camera_meta_edit_does_not_break_rig_open(tmp_path):
    """#71: open checks the camera THW, not the file size — a
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


# ---- review pass 7: legacy / adapters removed (#82-#84) ----
def test_82_open_requires_file_type(tmp_path):
    """A file without attrs.file_type is not an h5scene file (no legacy pass)."""
    import h5py
    from h5scene import H5Geo
    p = tmp_path / "c.h5"
    H5Cam.create(p, 1, 4, 4).close()
    with h5py.File(p, "r+") as h:
        del h.attrs["file_type"]
    with pytest.raises(ValueError, match="file_type"):
        H5Cam.open(p)
    g = tmp_path / "g.h5"
    H5Geo.create(g, np.zeros((3, 3), np.float32)).close()
    with h5py.File(g, "r+") as h:
        del h.attrs["file_type"]
    with pytest.raises(ValueError, match="file_type"):
        H5Geo.open(g)


def test_83_rgb_bytes_no_TN_adapter(tmp_path):
    """(T, N) image bytes / paths are rejected (the old rig layout)."""
    b = encode_rgb(np.zeros((8, 8, 3), np.uint8))
    a = np.empty((2, 2), object)
    a[:] = [[b, b], [b, b]]
    with H5Cam.create(tmp_path / "c.h5", 2, 8, 8) as cam:
        with pytest.raises(ValueError, match=r"flat \(T,\)"):
            cam.set_rgb_bytes(a)
        with pytest.raises(ValueError, match="frames but T"):
            cam.set_rgb_bytes([b] * 4)                # 2T frames: no silent [:T]
        with pytest.raises(ValueError):
            cam.set_rgb(np.zeros((2, 1, 8, 8, 3), np.uint8))   # (T,N,H,W,3)


def test_84_verify_dir_rejects_untagged(tmp_path):
    import h5py
    from h5scene.verify import verify_dir
    with h5py.File(tmp_path / "x.h5", "w") as h:
        h.create_dataset("a", data=[1])
    (r,) = verify_dir(str(tmp_path))
    assert not r["ok"] and "not an h5scene file" in r["checks"]["open"]["error"]


def _K33():
    return np.array([[4., 0., 3.], [0., 4., 3.], [0., 0., 1.]], np.float32)


def test_86_87_write_frame_rejects_wrong_units(tmp_path):
    """write_frame's depth unit guards match the set_* methods: a float is
    never depth_mm (it wraps/truncates), an int is never depth_meters
    (1500 'm' -> 0) or depth_conf, and a negative mm stores the invalid
    sentinel 0 (exactly like set_depth_mm), not a valid 65436."""
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 4, 8) as cam:
        cam.allocate(sections=("depth",))
        with pytest.raises(TypeError, match="meters"):
            cam.write_frame(0, depth_mm=np.full((4, 8), 1.5, np.float32))
        with pytest.raises(TypeError, match="integer MM"):
            cam.write_frame(0, depth_meters=np.full((4, 8), 1500, np.int32))
        # a negative mm stores the sentinel 0, not a valid 65436
        cam.write_frame(0, depth_mm=np.full((4, 8), -100, np.int32))
        assert (cam.depth_mm(0) == 0).all()
        cam.write_frame(1, depth_mm=np.full((4, 8), 200, np.uint16))
        cam.finalize()
    with H5Cam.create(str(tmp_path / "d.h5"), 1, 4, 8) as cam:
        cam.allocate(sections=("depth", "depth_conf"))
        with pytest.raises(TypeError, match="integers are not accepted"):
            cam.write_frame(0, depth_conf=np.full((4, 8), 1, np.int32))
        # nothing was poisoned: the frame is still writable
        cam.write_frame(0, depth_mm=np.full((4, 8), 100, np.uint16),
                        depth_conf=np.full((4, 8), 1.0, np.float32))
        cam.finalize()


def test_101_write_frame_frame_is_one_frame(tmp_path):
    """A (2, h, w) 'frame' is rejected — frame 1 used to be dropped silently."""
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 4, 8) as cam:
        cam.allocate(sections=("depth", "depth_conf"))
        with pytest.raises(ValueError, match=r"\(h, w\) or \(1, h, w\)"):
            cam.write_frame(0, depth_mm=np.full((2, 4, 8), 100, np.uint16))
        with pytest.raises(ValueError, match=r"\(h, w\) or \(1, h, w\)"):
            cam.write_frame(0, depth_conf=np.ones((2, 4, 8), np.float32))
        # one frame per call is fine (both (1, h, w) and (h, w))
        cam.write_frame(0, depth_mm=np.full((1, 4, 8), 100, np.uint16),
                        depth_conf=np.ones((4, 8), np.float32))
        cam.write_frame(1, depth_mm=np.full((4, 8), 100, np.uint16),
                        depth_conf=np.ones((1, 4, 8), np.float32))
        cam.finalize()


def test_88_write_frame_per_section_written_and_no_poison(tmp_path):
    """written is tracked PER SECTION (depth for t in one call, K in another
    is fine), and a call that fails part-way leaves t clean (a retry works)."""
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 4, 8) as cam:
        cam.allocate(sections=("depth", "K"))
        cam.write_frame(0, depth_mm=np.full((4, 8), 100, np.uint16))
        cam.write_frame(0, K=_K33())                     # a different section, same t
        with pytest.raises(ValueError, match="already written"):
            cam.write_frame(0, depth_mm=np.full((4, 8), 1, np.uint16))
        # a bad K in the same call must not poison the depth half
        badK = _K33(); badK[0, 0] = -4.
        with pytest.raises(ValueError, match="fx / fy"):
            cam.write_frame(1, depth_mm=np.full((4, 8), 1, np.uint16), K=badK)
        assert 1 not in cam._written["depth"]            # depth for t=1 not written
        cam.write_frame(1, depth_mm=np.full((4, 8), 1, np.uint16))
        cam.write_frame(1, K=_K33())
        cam.finalize()


def test_89_90_allocate_validates_all_first(tmp_path):
    """allocate validates EVERY section before it creates any dataset or sets
    the state (a mid-way failure leaves the file re-allocatable), and it
    rejects existing /img frames with a clear error."""
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 4, 8) as cam:
        with pytest.raises(ValueError, match="aspect ratio"):
            cam.allocate(sections=("depth", "K"), depth_hw=(4, 2))  # bad hw
        cam.allocate(sections=("depth",))          # still allocatable
        cam.close()
    with H5Cam.create(str(tmp_path / "i.h5"), 2, 4, 8) as cam:
        cam.set_rgb(np.zeros((2, 4, 8, 3), np.uint8))
        with pytest.raises(ValueError, match=r"del_section\('img'\)"):
            cam.allocate(sections=("img",))


def test_91_check_K_pose_keep_input_dtype(tmp_path):
    """set_K / set_pose_c2w store float32 (the SECTIONS dtype) — check_K /
    check_pose used to return a float64 copy, doubling the space and making
    cam.K() dtype depend on the writer."""
    from h5scene.codecs import check_K, check_pose
    k = _K33()
    assert check_K(k).dtype == np.float32            # the input, not a copy
    p = np.eye(4, dtype=np.float32)
    assert check_pose(p).dtype == np.float32
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 4, 8) as cam:
        cam.set_K(k)
        cam.set_pose_c2w(p)
    with h5py.File(str(tmp_path / "c.h5")) as h:
        assert h["K"].dtype == np.float32 and h["T"].dtype == np.float32


def test_92_93_set_meta_src_frame_ids_checked_and_cache_cleared(tmp_path):
    """The uniqueness check runs through set_meta too (both forms), and every
    write of src_frame_ids invalidates the index_of cache (a stale cache
    used to KeyError the new ids)."""
    with H5Cam.create(str(tmp_path / "c.h5"), 3, 4, 8) as cam:
        with pytest.raises(ValueError, match="unique"):
            cam.set_meta("src_frame_ids", ["a", "a", "b"])
        with pytest.raises(ValueError, match="unique"):
            cam.set_meta(src_frame_ids=["a", "a", "b"])
        cam.set_src_frame_ids(["a", "b", "c"])
        assert cam.index_of("b") == 1                # populate the cache
        cam.set_src_frame_ids(["x", "y", "z"])
        assert cam.index_of("z") == 2                # the cache was cleared
        with pytest.raises(KeyError):
            cam.index_of("b")


def test_94_check_K_pose_reject_nan_inf(tmp_path):
    """NaN/inf in cx / cy / the translation used to slip past the tolerance
    checks (abs(nan - 1) > eps is False) — they are rejected now."""
    from h5scene.codecs import check_K, check_pose
    k = _K33(); k[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        check_K(k)
    k = _K33(); k[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        check_K(k)
    p = np.eye(4); p[0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        check_pose(p)
    with H5Cam.create(str(tmp_path / "c.h5"), 1, 4, 8) as cam:
        with pytest.raises(ValueError, match="finite"):
            cam.set_K(k)
        p = np.eye(4); p[2, 3] = np.inf
        with pytest.raises(ValueError, match="finite"):
            cam.set_pose_c2w(p)


def test_95_verify_single_file_dispatches_on_file_type(tmp_path, capsys):
    """A single file argument verifies as ITS class (a rig/geo file used to
    be pushed through H5Cam-only verify_file and reported FAIL)."""
    from h5scene import H5Geo
    from h5scene.verify import main as verify_main, verify_any
    f = str(tmp_path / "g.geo.h5")
    H5Geo.create(f, np.zeros((4, 3), np.float32)).close()
    assert verify_any(f)["ok"] is True
    assert verify_main([f]) == 0 and "OK" in capsys.readouterr().out


def test_96_h5geo_rejects_slash_names_and_groups(tmp_path):
    """A '/' in a custom attribute name made a nested group that crashed the
    next open (AttributeError on a Group) — the write is rejected now, and
    open rejects non-dataset members with a clear error."""
    import h5py
    from h5scene import H5Geo
    f = str(tmp_path / "g.geo.h5")
    g = H5Geo.create(f, np.zeros((4, 3), np.float32))
    with pytest.raises(ValueError, match="must not contain"):
        g.set_point_attr("a/b", np.zeros(4, np.float32))
    g.close()
    with h5py.File(f, "r+") as h:                    # simulate the corrupt file
        h["points"].create_group("a/b")
    with pytest.raises(ValueError, match="not a dataset"):
        H5Geo.open(f)


# ---- review pass 7: performance / API (#99, #100) ----
def test_99_set_triangles_accepts_unsigned(tmp_path):
    """uint32 faces (Open3D / PLY readers) round-trip as the stored int32."""
    from h5scene import H5Geo
    g = H5Geo.create(tmp_path / "g.h5", np.zeros((12, 3), np.float32))
    g.set_triangles(np.arange(12, dtype=np.uint32).reshape(4, 3))
    tri = g.triangles()
    g.close()
    assert tri.dtype == np.int32
    assert tri.ravel().tolist() == list(range(12))
    # a float index is still rejected (an index is an integer kind only)
    g = H5Geo.create(tmp_path / "f.h5", np.zeros((12, 3), np.float32))
    with pytest.raises(TypeError, match="int/uint"):
        g.set_triangles(np.zeros((4, 3), np.float32))
    g.close()


def test_100_verify_uses_the_codec_row_checks(tmp_path):
    """verify_file's K/pose checks are codecs.pose's row error lists (no
    duplicate rule set). The writers reject the bad rows, so the file is
    corrupted through raw h5py: a NaN K (accepted by the old verify copy,
    whose abs(nan-1) > eps is False) and a non-orthonormal pose row must
    both fail."""
    from h5scene.codecs import K_row_errors, pose_row_errors
    from h5scene.verify import verify_file
    p = str(tmp_path / "c.h5")
    with H5Cam.create(p, 1, 8, 8) as cam:
        cam.set_K(np.array([[8., 0., 4.], [0., 8., 4.], [0., 0., 1.]],
                           np.float32))
        cam.set_pose_c2w(np.eye(4, dtype=np.float32))
    with h5py.File(p, "r+") as h:
        h["K"][0, 0, 0] = np.nan                        # the NaN row
        h["T"][0] = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0.5, 0, 0], [0, 0, 0, 1]], np.float32)
    r = verify_file(p)
    assert not r["checks"]["K"]["ok"]
    assert any("K frame 0" in e and "finite" in e
               for e in r["checks"]["K"]["errors"])
    assert not r["checks"]["pose"]["ok"]
    assert any("pose frame 0" in e for e in r["checks"]["pose"]["errors"])
    # the row lists are the shared implementation (an empty list = OK):
    K = np.array([[8., 0., 4.], [0., 8., 4.], [0., 0., 1.]], np.float64)
    assert K_row_errors(K[None]) == []
    assert pose_row_errors(np.eye(4, dtype=np.float64)[None]) == []


def test_98_verify_depth_reads_frame_by_frame(tmp_path):
    """verify_file's depth check aggregates per frame (not d[:] over the
    whole sequence): the report is identical to before for a multi-frame
    depth (valid min/max, the 65535 count, the valid fraction)."""
    from h5scene.verify import verify_file
    T, H, W = 6, 8, 8
    p = str(tmp_path / "c.h5")
    with H5Cam.create(p, T, H, W) as cam:
        cam.set_depth_mm(np.arange(T * H * W, dtype=np.uint16).reshape(T, H, W) % 7)
    r = verify_file(p)
    d = (np.arange(T * H * W, dtype=np.uint16) % 7).reshape(T, H, W)
    v = (d != 0)
    assert r["checks"]["depth"]["ok"] and r["checks"]["depth"]["n_65535"] == 0
    assert r["checks"]["depth"]["valid_min"] == int(d[v].min())
    assert r["checks"]["depth"]["valid_max"] == int(d[v].max())
    assert r["checks"]["depth"]["valid_frac"] == round(float(v.mean()), 6)
    # a raw 65535 in ONE frame is counted (the sentinel check survives the
    # per-frame loop)
    with h5py.File(p, "r+") as h:
        h["depth"][3, 2, 2] = 65535
    r = verify_file(p)
    assert r["checks"]["depth"]["n_65535"] == 1 and not r["ok"]


# ---- review pass 8 (#105–#113) ----

def test_105_write_frame_checks_shape_and_state_before_any_write(tmp_path):
    """A depth shape mismatch or an already-written section must fail BEFORE
    rgb is written (the frame stays untouched and retryable)."""
    rgb = np.zeros((8, 8, 3), np.uint8)
    with H5Cam.create(str(tmp_path / "a.h5"), 2, 8, 8) as cam:
        cam.allocate(("img", "depth"), depth_hw=(4, 4))
        with pytest.raises(ValueError, match="allocated"):
            cam.write_frame(0, rgb=rgb, depth_mm=np.ones((8, 8), np.uint16))
        assert "0" not in cam.h["img"] and cam._written["img"] == set()
        cam.write_frame(0, rgb=rgb, depth_mm=np.ones((4, 4), np.uint16))
        cam.write_frame(1, depth_mm=np.ones((4, 4), np.uint16))
        with pytest.raises(ValueError, match="already written"):
            cam.write_frame(1, rgb=rgb, depth_mm=np.ones((4, 4), np.uint16))
        assert 1 not in cam._written["img"]


def test_106_write_frame_K_pose_shape(tmp_path):
    with H5Cam.create(str(tmp_path / "a.h5"), 2, 8, 8) as cam:
        cam.allocate(("img", "K", "T"))
        for kw in ({"K": np.eye(4)}, {"K": np.stack([np.eye(3)] * 2)},
                   {"K": np.ones(3)}, {"pose_c2w": np.eye(3)}):
            with pytest.raises(ValueError, match="frame must be"):
                cam.write_frame(0, rgb=np.zeros((8, 8, 3), np.uint8), **kw)
        assert cam._written["img"] == set()


def test_107_write_frame_rgb_batch_rejected(tmp_path):
    with H5Cam.create(str(tmp_path / "a.h5"), 2, 8, 8) as cam:
        cam.allocate(("img",))
        with pytest.raises(ValueError, match="ONE"):
            cam.write_frame(0, rgb=np.zeros((2, 8, 8, 3), np.uint8))
        cam.write_frame(0, rgb=np.zeros((1, 8, 8, 3), np.uint8))


def test_108_index_of_cache_dropped_on_remove(tmp_path):
    with H5Cam.create(str(tmp_path / "a.h5"), 2, 8, 8) as cam:
        cam.set_src_frame_ids(["a", "b"])
        assert cam.index_of("a") == 0
        cam.set_meta("src_frame_ids", None)
        with pytest.raises(ValueError, match="no src_frame_ids"):
            cam.index_of("a")
        cam.set_meta(src_frame_ids=["x", "y"])       # multi-key form too
        assert cam.index_of("y") == 1


def test_109_duplicate_ids_message_is_bounded(tmp_path):
    with H5Cam.create(str(tmp_path / "a.h5"), 1000, 1, 1) as cam:
        with pytest.raises(ValueError) as e:
            cam.set_src_frame_ids([str(i // 2) for i in range(1000)])
        assert "500 duplicated" in str(e.value) and len(str(e.value)) < 200


def test_110_allocate_depth_hw_must_be_integers(tmp_path):
    with H5Cam.create(str(tmp_path / "a.h5"), 2, 8, 8) as cam:
        with pytest.raises(TypeError, match="depth_hw"):
            cam.allocate(("depth",), depth_hw=(4.0, 4.0))
        with pytest.raises(ValueError, match="depth_hw"):
            cam.allocate(("depth",), depth_hw=(4, 4, 1))
        cam.allocate(("depth",), depth_hw=(np.int64(4), 4))   # still allocatable


def test_111_del_section_refused_while_allocated(tmp_path):
    with H5Cam.create(str(tmp_path / "a.h5"), 2, 8, 8) as cam:
        cam.allocate(("depth",))
        with pytest.raises(ValueError, match="finalize"):
            cam.del_section("depth")


def test_112_verify_dir_uses_verify_any(tmp_path):
    from h5scene.verify import verify_dir
    (tmp_path / "junk.h5").write_bytes(b"junk")
    with h5py.File(tmp_path / "tagless.h5", "w"):
        pass
    r = {Path(x["path"]).name: x for x in verify_dir(str(tmp_path))}
    assert not r["junk.h5"]["ok"] and not r["tagless.h5"]["ok"]
    assert "not an h5scene file" in r["tagless.h5"]["checks"]["open"]["error"]


def test_113_peek_requires_is_metric(tmp_path):
    p = str(tmp_path / "a.h5")
    H5Cam.create(p, 2, 8, 8).close()
    assert H5Cam.peek(p)["is_metric"] is False
    with h5py.File(p, "r+") as h:
        del h.attrs["is_metric"]
    with pytest.raises(ValueError, match="is_metric"):
        H5Cam.peek(p)
