"""Tests for the h5scene package: write_scene_h5/verify_scene_h5 (functional
API) + H5Scene (direct file handle)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # h5scene/ (package root)
from h5scene import H5Scene, write_scene_h5, verify_scene_h5  # noqa: E402


def test_round_trip(tmp_path):
    T, N, H, W = 2, 2, 8, 16
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(T, N, H, W, 3)).astype(np.uint8)
    depth_m = rng.uniform(0.1, 30.0, size=(T, N, H, W)).astype(np.float32)
    depth_m[0, 0, 0, 0] = 70.0          # > 65.535 m -> must be stored 0
    K = np.array([[300., 0., W / 2], [0., 250., H / 2], [0., 0., 1.]], np.float32)
    Tpose = np.eye(4, dtype=np.float32)
    # multiples of 1/512 -> exactly representable in fp16, so the round-trip is exact
    conf01 = (rng.integers(512, 1024, size=(T, N, H, W)) / 1024.0).astype(np.float32)

    out = str(tmp_path / "scene.h5")
    got = write_scene_h5(out, rgb, depth=depth_m, K=K, pose=Tpose, depth_conf=conf01,
                         scene_path="a/b", meta_json='{"x": 1}', is_video=False)
    assert got == (T, N, H, W)

    ok, msgs = verify_scene_h5(out, rgb, depth=depth_m, K=K, pose=Tpose, depth_conf=conf01)
    assert ok, msgs

    import cv2
    import h5py
    with h5py.File(out, "r") as h:
        assert len(h["rgb"]) == T * N and "TNHW" in h.attrs
        assert tuple(int(x) for x in h.attrs["TNHW"]) == (T, N, H, W)
        for t in range(T):
            for n in range(N):
                dec = cv2.imdecode(h["rgb"][f"{t}_{n}"][:], cv2.IMREAD_COLOR)
                assert dec is not None and dec.shape == (H, W, 3)
        d = h["depth"][:]
        assert d.dtype == np.uint16 and d[0, 0, 0, 0] == 0   # 70 m -> clipped to invalid
        assert d.max() <= 65535 and (d[1:] > 0).all()
        assert tuple(h["K"].shape) == (T, N, 3, 3) and np.array_equal(h["K"][:], np.tile(K, (T, N, 1, 1)))
        assert tuple(h["T"].shape) == (T, N, 4, 4)
        c = h["depth_conf"][:]
        assert c.dtype == np.float16 and c.max() <= 1.0
        assert np.array_equal(c, conf01.astype(np.float16))
        assert h["scene_path"][()].decode() == "a/b"
        assert h["meta_json"][()].decode() == '{"x": 1}'
        assert not h["meta"]["is_video"][()] and h["meta"]["is_metric"][()]


def test_bytes_sequence(tmp_path):
    """Flat list of raw png bytes: stored verbatim (bytes in == bytes out)."""
    import cv2
    import h5py
    frames = []
    for i in range(3):
        img = np.full((8, 16, 3), i * 40, np.uint8)
        frames.append(cv2.imencode(".png", img)[1].tobytes())
    out = str(tmp_path / "seq.h5")
    got = write_scene_h5(out, frames)
    assert got == (3, 1, 8, 16)
    ok, msgs = verify_scene_h5(out, frames)
    assert ok, msgs
    with h5py.File(out, "r") as h:
        for i, b in enumerate(frames):
            assert np.array_equal(h["rgb"][f"{i}_0"][:], np.frombuffer(b, np.uint8))


def test_mask(tmp_path):
    """mask: per-pixel (T,N,H,W), broadcast (H,W), and per-slot (T,N)."""
    import h5py
    T, N, H, W = 2, 2, 4, 8
    rgb = np.zeros((T, N, H, W, 3), np.uint8)
    pixel_mask = np.ones((T, N, H, W), bool)
    pixel_mask[1, 0, 2:, :] = False          # second frame of cam0: bottom half padding
    out = str(tmp_path / "m1.h5")
    write_scene_h5(out, rgb, mask=pixel_mask)
    ok, msgs = verify_scene_h5(out, rgb, mask=pixel_mask)
    assert ok, msgs
    with h5py.File(out, "r") as h:
        assert h["mask"].dtype == bool and np.array_equal(h["mask"][:], pixel_mask)

    # (H,W) broadcast
    hm = np.ones((H, W), bool)
    hm[:, -2:] = False
    out = str(tmp_path / "m2.h5")
    write_scene_h5(out, rgb, mask=hm)
    assert verify_scene_h5(out, rgb, mask=hm)[0]
    with h5py.File(out, "r") as h:
        assert np.array_equal(h["mask"][:], np.broadcast_to(hm, (T, N, H, W)))

    # (T,N) per-slot: frame (1,1) is padding
    slot = np.ones((T, N), bool)
    slot[1, 1] = False
    out = str(tmp_path / "m3.h5")
    write_scene_h5(out, rgb, mask=slot)
    assert verify_scene_h5(out, rgb, mask=slot)[0]
    with h5py.File(out, "r") as h:
        m = h["mask"][:]
        assert m[1, 1].sum() == 0 and m[1, 0].all() and m[0].all()


def test_single_frame_no_optional(tmp_path):
    H, W = 4, 6
    rgb = np.zeros((H, W, 3), np.uint8)
    out = str(tmp_path / "one.h5")
    got = write_scene_h5(out, rgb)
    assert got == (1, 1, H, W)
    ok, msgs = verify_scene_h5(out, rgb)
    assert ok, msgs
    import h5py
    with h5py.File(out, "r") as h:
        assert len(h["rgb"]) == 1
        assert "depth" not in h and "K" not in h and "T" not in h and "depth_conf" not in h


def test_depth_mask(tmp_path):
    """depth_mask: derived (not stored) — depth valid (0 AND 65535 both excluded)
    and, when a depth_conf section exists, conf > 0 (or > conf_min when given)."""
    H, W = 4, 8
    rgb = np.zeros((H, W, 3), np.uint8)
    depth = np.full((H, W), 500, np.uint16)
    depth[0, 0] = 0        # invalid sentinel 0
    depth[1, 1] = 65535    # invalid sentinel (also written 0, but check the read path)
    out = str(tmp_path / "dm.h5")
    write_scene_h5(out, rgb, depth=depth)
    with H5Scene(out) as s:
        m = s.depth_mask(0, 0)
        assert m.dtype == bool and m.shape == (H, W)
        # no conf section: only the two sentinels excluded, the rest valid
        assert not m[0, 0] and not m[1, 1]
        assert m.sum() == (H * W - 2)
    # with a depth_conf: conf 0 also excludes; conf_min narrows the gate
    conf = np.ones((H, W), np.float32)
    conf[0, 1] = 0.0
    conf[1, 0] = 0.5
    out = str(tmp_path / "dm2.h5")
    write_scene_h5(out, rgb, depth=depth, depth_conf=conf)
    with H5Scene(out) as s:
        m = s.depth_mask(0, 0)
        assert not m[0, 1]                        # conf 0 -> excluded
        assert m[1, 0]                            # conf 0.5 -> passes the default gate
        assert not s.depth_mask(0, 0, conf_min=0.5)[1, 0]   # strict > : conf 0.5 excluded
        assert not s.depth_mask(0, 0, conf_min=0.6)[1, 0]
    # not stored as a section
    import h5py
    with h5py.File(out, "r") as h:
        assert "depth_mask" not in h


def test_h5scene_file_handle(tmp_path):
    """H5Scene(path, T,N,H,W): create + set_* write straight into the file;
    re-open reads straight back. No in-memory state, no save()."""
    T, N, H, W = 2, 2, 8, 16
    rng = np.random.default_rng(1)
    # jpg q95 is lossy: a smooth ramp keeps the decoded round-trip close (max err 13 on 8x16),
    # so assert a bounded error instead of byte equality
    X = np.tile(np.linspace(0, 255, W), (H, 1))
    _rgb0 = np.stack([X, 255 - X, np.full((H, W), 128.0)], axis=2).astype(np.uint8)
    rgb = np.repeat(_rgb0, T * N, axis=0).reshape(T, N, H, W, 3)
    depth_m = rng.uniform(0.1, 30.0, size=(T, N, H, W)).astype(np.float32)
    depth_m[0, 0, 0, 0] = 70.0
    K = np.array([[300., 0., W / 2], [0., 250., H / 2], [0., 0., 1.]], np.float32)
    pose = np.eye(4, dtype=np.float32)
    mask = np.ones((T, N, H, W), bool)
    mask[1, 1, :, :3] = False

    out = str(tmp_path / "s.h5")
    s = H5Scene(out, T, N, H, W)
    assert s.shape == (T, N, H, W)
    assert s.keys == [f"{t}_{n}" for t in range(T) for n in range(N)]
    assert s.is_metric and s.is_video
    assert not s.has("rgb") and not s.has("depth")
    try:
        # setters write directly; reads reflect the file immediately
        s.set_rgb(rgb)
        assert s.has("rgb")
        assert np.abs(s.rgb(0, 0).astype(int) - rgb[0, 0].astype(int)).max() <= 40  # jpg q95 lossy
        s.set_depth(depth_m)
        d = s.depth(0, 0)
        assert d.dtype == np.uint16 and d[0, 0] == 0  # 70 m -> invalid
        mm = np.rint(depth_m[1, 1] * 1000).astype(np.int32).clip(0, 65535)
        assert np.array_equal(s.depth(1, 1, meters=True), (mm / 1000.0).astype(np.float32))
        s.set_mask(mask)
        assert s.mask(1, 1)[:, :3].sum() == 0 and s.mask(1, 1)[:, 3:].all()
        s.set_K(K)
        s.set_pose(pose)
        assert np.array_equal(s.K(0, 0), K) and np.array_equal(s.pose(0, 0), pose)
        assert np.array_equal(s.K(1, 1), K)
        s.set_meta(scene_path="x/y", meta_json='{"a": 2}')
        assert s.scene_path == "x/y" and s.meta_json == '{"a": 2}'
    finally:
        s.close()

    # re-open: reads straight from the file, everything consistent
    with H5Scene(out) as s2:
        assert s2.shape == (T, N, H, W)
        for t in range(T):
            for n in range(N):
                assert np.abs(s2.rgb(t, n).astype(int) - rgb[t, n].astype(int)).max() <= 40
        assert s2.depth(0, 0)[0, 0] == 0
        assert np.array_equal(s2.mask(1, 1), mask[1, 1])
        assert s2.scene_path == "x/y" and s2.meta_json == '{"a": 2}'

    # overwrite via create-mode reuse; set_* replaces existing sections
    with H5Scene(out, T, N, H, W, is_video=False) as s3:
        assert not s3.has("rgb")            # create wipes the old file
        s3.set_rgb(rgb)
        s3.set_mask(np.zeros((T, N, H, W), bool))  # replace the previous mask
        assert not s3.mask(0, 0).any()
        assert s3.has("mask") and not s3.has("depth")


def test_h5scene_set_validation(tmp_path):
    T, N, H, W = 2, 2, 8, 16
    out = str(tmp_path / "v.h5")
    with H5Scene(out, T, N, H, W) as s:
        s.set_rgb(np.zeros((T, N, H, W, 3), np.uint8))
        bad = (
            (lambda: s.set_depth(np.zeros((T, N, H, W + 1), np.uint16)), "depth"),
            (lambda: s.set_mask(np.ones((H, W + 1), bool)), "mask"),
            (lambda: s.set_K(np.zeros((T, N, 3, 4), np.float32)), "K"),
            (lambda: s.set_pose(np.zeros((T, N + 1, 4, 4), np.float32)), "pose"),
            (lambda: s.set_depth_conf(np.zeros((T, N, H, W + 1), np.float32)), "depth_conf"),
        )
        for fn, name in bad:
            try:
                fn()
                raise AssertionError(f"set_{name} should have raised")
            except ValueError as e:
                assert name in str(e)
        s.set_mask(np.ones((T, N, H, W), bool))
        for t, n in ((T, 0), (0, N), (-1, 0)):
            try:
                s.mask(t, n)
                raise AssertionError("out-of-range (t,n) should have raised")
            except IndexError:
                pass
        # missing section -> KeyError
        for fn in (lambda: s.depth(0, 0), lambda: s.K(0, 0),
                   lambda: s.pose(0, 0), lambda: s.depth_conf(0, 0)):
            try:
                fn()
                raise AssertionError("expected KeyError")
            except KeyError:
                pass
    # create mode requires all four dims
    try:
        H5Scene(str(tmp_path / "bad.h5"), 2)
        raise AssertionError("create mode needs T,N,H,W")
    except ValueError:
        pass


def test_h5scene_absent_sections(tmp_path):
    T, N, H, W = 2, 1, 4, 6
    arr = np.stack([np.full((H, W, 3), i, np.uint8) for i in range(T)], axis=0)[:, None]
    assert arr.shape == (T, N, H, W, 3)
    out = str(tmp_path / "bare.h5")
    with H5Scene(out, T, N, H, W) as s:
        s.set_rgb(arr)
    with H5Scene(out) as s2:
        assert s2.shape == (T, N, H, W)
        for sec in ("depth", "mask", "K", "T", "depth_conf"):
            assert not s2.has(sec)
            fn = {"depth": s2.depth, "mask": s2.mask, "K": s2.K,
                  "T": s2.pose, "depth_conf": s2.depth_conf}[sec]
            try:
                fn(0, 0)
                raise AssertionError("expected KeyError")
            except KeyError:
                pass
        for i in range(T):
            assert np.array_equal(s2.rgb(i, 0), arr[i, 0])


if __name__ == "__main__":
    test_round_trip(Path(tempfile.mkdtemp()))
    test_bytes_sequence(Path(tempfile.mkdtemp()))
    test_mask(Path(tempfile.mkdtemp()))
    test_single_frame_no_optional(Path(tempfile.mkdtemp()))
    test_h5scene_file_handle(Path(tempfile.mkdtemp()))
    test_h5scene_set_validation(Path(tempfile.mkdtemp()))
    test_h5scene_absent_sections(Path(tempfile.mkdtemp()))
    print("OK")
