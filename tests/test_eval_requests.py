"""Tests for the requirements_eval requests E1-E8 + the attrs.file_type
tag (2026-11-13; the E-numbering follows requirements_eval/REQUIREMENTS.md)."""
import cv2
import numpy as np
import pytest
import h5py

from h5scene import H5Cam, H5CamRig
from h5scene.codecs import check_K, check_pose
from h5scene.verify import verify_file, verify_dir, main as verify_main


def _K(h, w):
    return np.array([[w, 0., w / 2], [0., h, h / 2], [0., 0., 1.]], np.float32)


def _cam_with(tmp_path, T, H, W, name="c.h5"):
    cam = H5Cam.create(str(tmp_path / name), T, H, W)
    cam.set_rgb(np.full((T, H, W, 3), 5, np.uint8))
    cam.set_K(_K(H, W))
    cam.set_pose_c2w(np.eye(4, dtype=np.float32))
    return cam


# ---- E1: string frame ids ---------------------------------------------------
def test_src_frame_ids_roundtrip_and_lookup(tmp_path):
    """E1: tum-style timestamp fids + dl3dv-style path fids round-trip exact;
    index_of / indices_of answer in order; unknown fid is a KeyError."""
    tum = ["1341846313.553992", "1341846313.619977", "1341846313.685972"]
    dl3dv = ["001dcc/images_8/frame_00001.png", "001dcc/images_8/frame_00002.png",
             "001dcc/images_8/frame_00003.png"]
    for fids in (tum, dl3dv):
        with _cam_with(tmp_path, 3, 8, 8) as cam:
            cam.set_src_frame_ids(fids)
        with H5Cam.open(str(tmp_path / "c.h5")) as cam:
            assert cam.src_frame_ids == fids
            assert cam.index_of(fids[1]) == 1
            assert cam.indices_of([fids[2], fids[0]]) == [2, 0]
            with pytest.raises(KeyError):
                cam.index_of("nope")
    # a file without ids: None / a clear error
    with H5Cam.create(str(tmp_path / "p.h5"), 3, 8, 8) as cam:
        pass
    with H5Cam.open(str(tmp_path / "p.h5")) as cam:
        assert cam.src_frame_ids is None
        with pytest.raises(ValueError, match="no src_frame_ids"):
            cam.index_of("x")


def test_src_frame_ids_guarded(tmp_path):
    """E1: non-strings raise (int-like -> the src_frame_idxs name), wrong
    length and duplicates raise; they can coexist with src_frame_idxs."""
    with _cam_with(tmp_path, 3, 8, 8) as cam:
        with pytest.raises(TypeError, match="strings"):
            cam.set_src_frame_ids([0, 1, 2])
        with pytest.raises(ValueError, match="length"):
            cam.set_src_frame_ids(["a", "b"])
        with pytest.raises(ValueError, match="unique"):
            cam.set_src_frame_ids(["a", "b", "a"])
        cam.set_src_frame_ids(["a", "b", "c"])
        cam.set_meta(src_frame_idxs=np.array([5, 9, 12]))
    with H5Cam.open(str(tmp_path / "c.h5")) as cam:
        assert cam.src_frame_ids == ["a", "b", "c"]
        assert list(cam.src_frame_idxs) == [5, 9, 12]
    # the rig / geo registries do NOT inherit the T-bound frame keys
    assert "src_frame_ids" not in H5CamRig.META_KEYS
    assert "timestamps" not in H5CamRig.META_KEYS
    assert "src_frame_idxs" not in H5CamRig.META_KEYS


# ---- E3: the K / pose convention checks --------------------------------------
def test_K_and_pose_convention_checks(tmp_path):
    """E3: a pinhole K and a c2w pose pass; each bad input raises."""
    assert check_K(_K(8, 8)).shape == (3, 3)
    assert check_pose(np.eye(4)).shape == (4, 4)
    for m in (
            np.array([[0., 0., 4.], [0., 8., 4.], [0., 0., 1.]]),   # fx <= 0
            np.array([[8., 0., 4.], [0., 8., 4.], [0., 0., 2.]]),   # K[2,2]
            np.array([[8., 0., 4.], [1., 8., 4.], [0., 0., 1.]]),   # skew K[1,0]
            np.array([[8., 0., 4.], [0., 8., 4.], [0., 0.1, 1.]])):  # K[2,1]
        with pytest.raises(ValueError, match="K frame"):
            check_K(m)
    # a (T, 3, 3) batch with one bad row: the frame index is reported
    good = _K(8, 8)
    with pytest.raises(ValueError, match="frame 2"):
        check_K(np.stack([good, good, np.array([[8., 0., 4.], [0., 8., 4.], [0., 0., 2.]])]))
    for m in (
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 1]]),
            np.array([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])):
        with pytest.raises(ValueError, match="pose frame"):
            check_pose(m)
    # the writers enforce it too
    with _cam_with(tmp_path, 2, 8, 8) as cam:
        with pytest.raises(ValueError, match="K frame"):
            cam.set_K(np.array([[-8., 0., 4.], [0., 8., 4.], [0., 0., 1.]], np.float32))
        with pytest.raises(ValueError, match="pose frame"):
            cam.set_pose_c2w(np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                       [0, 0.5, 0, 0], [0, 0, 0, 1]], np.float32))


# ---- E4: the encoded-bytes reader --------------------------------------------
def test_img_bytes_roundtrip(tmp_path):
    """E4: cv2.imdecode(img_bytes(t)) (as RGB) equals rgb(t); the stored
    bytes ARE the stored jpg."""
    with _cam_with(tmp_path, 2, 8, 8) as cam:
        cam.set_rgb(np.full((2, 8, 8, 3), 7, np.uint8))
    with H5Cam.open(str(tmp_path / "c.h5")) as cam:
        b = cam.img_bytes(1)
        assert isinstance(b, bytes) and b[:2] == b"\xff\xd8"
        dec = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        assert (dec == cam.rgb(1)).all()
    # a camera without img frames raises a KeyError
    with H5Cam.create(str(tmp_path / "n.h5"), 2, 8, 8) as cam:
        pass
    with H5Cam.open(str(tmp_path / "n.h5")) as cam:
        with pytest.raises(KeyError, match="img"):
            cam.img_bytes(0)


# ---- E5: per-frame / streaming write ------------------------------------------
def test_streaming_write_equals_batch(tmp_path):
    """E5: a 1000-frame scene written frame-by-frame equals the set_* version
    byte-for-byte; a skipped t makes finalize raise."""
    T, H, W = 1000, 4, 8
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (T, H, W, 3)).astype(np.uint8)
    depth = rng.integers(100, 3000, (T, H, W)).astype(np.uint16)
    # the batch version
    with H5Cam.create(str(tmp_path / "batch.h5"), T, H, W) as cam:
        cam.set_rgb(rgb)
        cam.set_depth_mm(depth)
    # the streaming version
    with H5Cam.create(str(tmp_path / "stream.h5"), T, H, W) as cam:
        cam.allocate(sections=("img", "depth"))
        for t in range(T):
            cam.write_frame(t, rgb=rgb[t], depth_mm=depth[t])
        cam.finalize()
    # byte-for-byte on the stored jpgs + the depth dataset (the acceptance:
    # the frame-by-file write equals the set_* version)
    with h5py.File(str(tmp_path / "batch.h5")) as a, h5py.File(str(tmp_path / "stream.h5")) as b:
        for t in (0, T - 1):
            assert (a["img"][f"{t}"][:] == b["img"][f"{t}"][:]).all()
        assert (a["depth"][:] == b["depth"][:]).all()
    # a skipped t raises at finalize
    with H5Cam.create(str(tmp_path / "skip.h5"), 3, 4, 8) as cam:
        cam.allocate(sections=("depth",))
        for t in (0, 2):
            cam.write_frame(t, depth_mm=np.full((4, 8), 100, np.uint16))
        with pytest.raises(ValueError, match="frames not written"):
            cam.finalize()
    # a double write is a clear error
    with H5Cam.create(str(tmp_path / "dbl.h5"), 2, 4, 8) as cam:
        cam.allocate(sections=("depth",))
        cam.write_frame(0, depth_mm=np.full((4, 8), 100, np.uint16))
        with pytest.raises(ValueError, match="already written"):
            cam.write_frame(0, depth_mm=np.full((4, 8), 100, np.uint16))


def test_streaming_write_frame_rgb_bytes(tmp_path):
    """E5 (review): write_frame accepts jpg BYTES (bytes AND bytearray) and
    creates /img itself when the file has no img group yet (r+ file)."""
    H, W = 4, 8
    rgb = np.full((H, W, 3), 200, np.uint8)
    ok = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                      [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    with H5Cam.create(str(tmp_path / "c.h5"), 2, H, W) as cam:
        cam.allocate(sections=("img",))            # no /img group written yet
        cam.write_frame(0, rgb=ok)                 # bytes -> decoded frame
        cam.write_frame(1, rgb=bytearray(ok))     # bytearray, not a uint8 array
        cam.finalize()
    with H5Cam.open(str(tmp_path / "c.h5")) as cam:
        assert (cam.rgb(0) == rgb).all() and (cam.rgb(1) == rgb).all()


# ---- E6: the verify command ----------------------------------------------------
def test_verify_file_and_main(tmp_path, capsys):
    """E6: a good file verifies clean (exit 0); a depth with raw 65535 fails
    (exit 1); the K check fails on a non-pinhole K; --json writes the report."""
    with _cam_with(tmp_path, 2, 8, 8) as cam:
        cam.set_depth_mm(np.full((2, 8, 8), 1000, np.uint16))
        cam.set_src_frame_ids(["a", "b"])
    r = verify_file(str(tmp_path / "c.h5"))
    assert r["ok"] and r["checks"]["depth"]["n_65535"] == 0
    assert r["checks"]["K"]["ok"] and r["checks"]["pose"]["ok"] and r["checks"]["img"]["ok"]
    assert r["checks"]["src_frame_ids"]["ok"]
    # a raw 65535 (bypassing the API: the writers map >= max to 0) must fail
    with h5py.File(str(tmp_path / "c.h5"), "r+") as h:
        h["depth"][0, 0, 0] = 65535
    assert verify_file(str(tmp_path / "c.h5"))["checks"]["depth"]["n_65535"] == 1
    assert not verify_file(str(tmp_path / "c.h5"))["ok"]
    # the CLI: exit 1 + the json report
    with h5py.File(str(tmp_path / "c.h5"), "r+") as h:
        h["depth"][0, 0, 0] = 1000
    rc = verify_main([str(tmp_path / "c.h5"), "--json", str(tmp_path / "r.json")])
    assert rc == 0
    import json
    assert json.load(open(str(tmp_path / "r.json")))[0]["ok"]
    out = capsys.readouterr().out
    assert "OK  " in out
    # a directory: *.h5 scanned (the non-H5Cam files open-checked)
    assert [r["path"].split("/")[-1] for r in verify_dir(str(tmp_path))]


# ---- E7: the light metadata read ------------------------------------------------
def test_peek(tmp_path):
    """E7: peek returns the listing fields WITHOUT the full open validation
    (a section-shape corruption still peeks; a non-cam file raises)."""
    with _cam_with(tmp_path, 3, 8, 8) as cam:
        cam.set_depth_mm(np.full((3, 8, 8), 1000, np.uint16))
        cam.set_meta(dataset="d", scene="s", seq="q")
        cam.set_src_frame_ids(["a", "b", "c"])
    d = H5Cam.peek(str(tmp_path / "c.h5"))
    assert d["file_type"] == "h5cam" and d["THW"] == (3, 8, 8)
    assert set(d["sections"]) == {"depth", "img", "K", "T"}
    assert d["dataset"] == "d" and d["scene"] == "s" and d["seq"] == "q"
    assert d["src_frame_ids"] == ["a", "b", "c"] and d["split"] is None
    with pytest.raises(FileNotFoundError):
        H5Cam.peek(str(tmp_path / "missing.h5"))
    # a non-H5Cam file is a clear error (its tag / THW is checked first)
    with h5py.File(str(tmp_path / "other.h5"), "w") as h:
        h.attrs["THW"] = np.array([3, 8, 8], np.int64)
    with pytest.raises(ValueError, match="not an H5Cam"):
        H5Cam.peek(str(tmp_path / "other.h5"))


# ---- E8: the float meta arrays ----------------------------------------------------
def test_timestamps_roundtrip(tmp_path):
    """E8: timestamps are a float64 (T,) meta key (int input is upcast, a
    float input is kept, a non-numeric element is rejected)."""
    with _cam_with(tmp_path, 3, 8, 8) as cam:
        with pytest.raises(TypeError, match="float"):
            cam.set_meta(timestamps=np.array(["a", "b", "c"]))
        cam.set_meta(timestamps=np.array([1341846313.553992, 1341846313.62, 1341846313.69]))
    with H5Cam.open(str(tmp_path / "c.h5")) as cam:
        ts = cam.timestamps
        assert ts.dtype == np.float64 and ts.shape == (3,)
        assert ts[0] == pytest.approx(1341846313.553992)


# ---- the attrs.file_type tag ----------------------------------------------------
def test_file_type_attr(tmp_path):
    """Every h5scene file pins attrs.file_type at create/build; the
    constructors reject a mismatched tag (the file is identifiable by type)."""
    with _cam_with(tmp_path, 2, 8, 8) as cam:
        pass
    with h5py.File(str(tmp_path / "c.h5"), "r") as h:
        assert h.attrs["file_type"] == "h5cam"
    # rig (the cameras must share dataset + scene)
    with H5Cam.open(str(tmp_path / "c.h5"), "r+") as cam:
        cam.set_meta(dataset="d", scene="s")
    a = str(tmp_path / "c.h5")
    with H5Cam.create(str(tmp_path / "d.h5"), 2, 8, 8) as cam:
        cam.set_rgb(np.full((2, 8, 8, 3), 9, np.uint8))
        cam.set_meta(dataset="d", scene="s")
        cam.set_K(_K(8, 8))
        cam.set_pose_c2w(np.eye(4, dtype=np.float32))
    rig = H5CamRig.build(str(tmp_path / "rig.h5"), [a, str(tmp_path / "d.h5")])
    with h5py.File(str(tmp_path / "rig.h5"), "r") as h:
        assert h.attrs["file_type"] == "h5camrig"
    rig.close()
    # a mismatched tag is rejected (a cam file claiming to be a rig)
    with h5py.File(str(tmp_path / "c.h5"), "r+") as h:
        h.attrs["file_type"] = "h5camrig"
    with pytest.raises(ValueError, match="file_type"):
        H5CamRig.open(str(tmp_path / "c.h5"))
