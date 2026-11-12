"""Tests for the h5scene package: H5Scene (direct file handle) + one-shot
write/verify helpers (test-local: the library only exposes H5Scene)."""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # h5scene/ (package root)
from h5scene import H5Scene  # noqa: E402
from h5scene.convert import (_broadcast, _to_depth_conf, _to_depth_mm, _to_mask)  # noqa: E402
from h5scene.rgb import _normalize_rgb  # noqa: E402


def write_scene_h5(out_path, rgb, depth=None, mask=None, K=None, pose=None, depth_conf=None,
                   dataset=None, scene=None, seq=None, suffix=None, split=None,
                   meta_json=None, frame_indices=None, tags=None, is_metric=True):
    """One-shot (test-only): create a scene and write every given section."""
    t, n, h, w, _f = _normalize_rgb(rgb)
    s = H5Scene(out_path, t, n, h, w, is_metric=is_metric)
    try:
        s.set_rgb(rgb)
        if depth is not None:
            s.set_depth(depth)
        if mask is not None:
            s.set_mask(mask)
        if K is not None:
            s.set_K(K)
        if pose is not None:
            s.set_pose(pose)
        if depth_conf is not None:
            s.set_depth_conf(depth_conf)
        if any(v is not None for v in (dataset, scene, seq, suffix, split,
                                       meta_json, frame_indices, tags)):
            s.set_meta(dataset=dataset, scene=scene, seq=seq, suffix=suffix,
                       split=split, meta_json=meta_json,
                       frame_indices=frame_indices, tags=tags)
    finally:
        s.close()
    return t, n, h, w


def verify_scene_h5(out_path, rgb, depth=None, mask=None, K=None, pose=None, depth_conf=None):
    """One-shot (test-only): re-read the h5 and byte-compare every section
    against the inputs (stored depth/depth_conf at their stored (h, w))."""
    ok, msgs = True, []

    def fail(msg):
        nonlocal ok
        ok = False
        msgs.append(msg)

    T, N, H, W, frames = _normalize_rgb(rgb)
    with H5Scene(out_path) as s:
        if s.shape != (T, N, H, W):
            fail(f"stored shape {s.shape} != input {(T, N, H, W)}")
        for i in range(T * N):
            t, n = divmod(i, N)
            fid = f"{t}_{n}"
            try:
                got = s.rgb(t, n)                      # decoded (H, W, 3) uint8
            except KeyError:
                fail(f"rgb {fid} missing")
                continue
            ref = cv2.imdecode(np.frombuffer(frames[i], np.uint8), cv2.IMREAD_COLOR)
            if got.shape != ref.shape:
                fail(f"rgb {fid} shape {got.shape} != {ref.shape}")
            elif np.abs(got.astype(int) - ref.astype(int)).max() > 40:
                fail(f"rgb {fid} pixels deviate > 40")
        for name, check, expected in (
            ("depth", lambda d: _to_depth_mm(d, T, N), depth),
            ("mask", lambda m: _to_mask(m, T, N, H, W), mask),
            ("K", lambda k: _broadcast(k, np.float32, T, N, (3, 3)), K),
            ("T", lambda t: _broadcast(t, np.float32, T, N, (4, 4)), pose),
            ("depth_conf", lambda c: _to_depth_conf(c, T, N), depth_conf),
        ):
            if expected is None:
                if s.has(name):
                    fail(f"{name} section present but input is None")
                continue
            if not s.has(name) or not np.array_equal(s._h[name][:], check(expected)):
                fail(f"{name} mismatch")
    return ok, msgs


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
                         dataset="7scenes", scene="a", seq="b",
                         split="test",
                         frame_indices=[112, 125],       # (T,); == arange(T) would be omitted
                         tags={"data_type": "real", "environment": "indoor"},
                         meta_json='{"x": 1}')
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
        m = h["meta"]
        assert m["dataset"][()].decode() == "7scenes"
        assert m["scene"][()].decode() == "a"
        assert m["seq"][()].decode() == "b"
        assert "suffix" not in m
        assert m["split"][()].decode() == "test"
        assert m["meta_json"][()].decode() == '{"x": 1}'
        assert np.array_equal(m["frame_indices"][:], np.array([112, 125]))
        assert m["tags"]["data_type"][()].decode() == "real"
        assert m["tags"]["environment"][()].decode() == "indoor"
        assert "is_metric" in h.attrs and bool(h.attrs["is_metric"])      # attrs (data-side)
        assert "is_metric" not in m                                       # legacy location cleared
        assert "is_video" not in m


def test_bytes_sequence(tmp_path):
    """jpg q95 frames stored verbatim (bytes in == bytes out); png re-encoded to jpg."""
    import cv2
    import h5py
    jpg95, png = [], []
    for i in range(3):
        img = np.full((8, 16, 3), i * 40, np.uint8)
        jpg95.append(cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes())
        png.append(cv2.imencode(".png", img)[1].tobytes())
    out = str(tmp_path / "seq.h5")
    got = write_scene_h5(out, jpg95)
    assert got == (3, 1, 8, 16)
    assert verify_scene_h5(out, jpg95)[0]
    with h5py.File(out, "r") as h:
        for i, b in enumerate(jpg95):  # q95 stored verbatim
            assert np.array_equal(h["rgb"][f"{i}_0"][:], np.frombuffer(b, np.uint8))
    # png frames: detected non-jpg -> re-encoded to jpg q95 (not stored verbatim)
    out = str(tmp_path / "seq_png.h5")
    write_scene_h5(out, png)
    with h5py.File(out, "r") as h:
        b0 = bytes(h["rgb"]["0_0"][:])
        assert b0[0] == 0xFF and b0[1] == 0xD8, "png frame was not re-encoded to jpg"
        assert b0 != png[0]


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


# ---- rgb path input + jpg q95 policy ----------------------------------------

def _write_png(tmp_path, img, name):
    import cv2
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return p


def test_rgb_path_input(tmp_path):
    """rgb accepts file paths (flat: T frames N=1; (T,N): N cameras per frame)."""
    import cv2
    H, W = 16, 32
    a = cv2.imread(str(_write_png(tmp_path, np.full((H, W, 3), 30, np.uint8), "a.png")))
    b = cv2.imread(str(_write_png(tmp_path, np.full((H, W, 3), 60, np.uint8), "b.png")))
    assert a is not None and b is not None
    out = str(tmp_path / "paths.h5")
    # (T, N) grid: 2 frames x 2 cams
    grid = np.array([[str(tmp_path / "a.png"), str(tmp_path / "b.png")],
                     [str(tmp_path / "a.png"), str(tmp_path / "b.png")]], dtype=object)
    got = write_scene_h5(out, grid)
    assert got == (2, 2, H, W)
    assert verify_scene_h5(out, grid)[0]
    # flat: T frames, N=1
    out = str(tmp_path / "flat.h5")
    assert write_scene_h5(out, [str(tmp_path / "a.png"), str(tmp_path / "b.png")]) == (2, 1, H, W)
    # single path string: 1 frame, N=1
    out = str(tmp_path / "one.h5")
    assert write_scene_h5(out, str(tmp_path / "a.png")) == (1, 1, H, W)


def test_rgb_jpg_quality_policy(tmp_path):
    """stored bytes are normalized to jpg q<=95:
    q95 stored verbatim, q100 re-encoded to q95, png re-encoded to jpg."""
    import cv2
    import h5py
    img = np.random.default_rng(0).integers(0, 256, (24, 40, 3)).astype(np.uint8)
    q95 = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
    q100 = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tobytes()
    png = cv2.imencode(".png", img)[1].tobytes()

    # q95 -> verbatim
    out = str(tmp_path / "q95.h5")
    write_scene_h5(out, [q95])
    with h5py.File(out, "r") as h:
        assert np.array_equal(h["rgb"]["0_0"][:], np.frombuffer(q95, np.uint8))

    # q100 -> re-encoded to q95 (not verbatim, and smaller than the q100 source)
    out = str(tmp_path / "q100.h5")
    write_scene_h5(out, [q100])
    with h5py.File(out, "r") as h:
        stored = bytes(h["rgb"]["0_0"][:])
        assert stored != q100 and stored != q95
        assert stored[:3] == bytes((0xFF, 0xD8, 0xFF))  # still a jpeg

    # png -> re-encoded to jpg
    out = str(tmp_path / "png.h5")
    write_scene_h5(out, [png])
    with h5py.File(out, "r") as h:
        assert bytes(h["rgb"]["0_0"][:])[:2] == bytes((0xFF, 0xD8))


def test_rgb_mismatched_resolution(tmp_path):
    """OCR#4: a later frame with a different size is rejected, not silently stored."""
    import cv2
    H, W = 16, 32
    a = str(_write_png(tmp_path, np.zeros((H, W, 3), np.uint8), "a.png"))
    b = str(_write_png(tmp_path, np.zeros((H, W * 2, 3), np.uint8), "b.png"))
    out = str(tmp_path / "bad.h5")
    with pytest.raises(ValueError, match=r"size"):
        write_scene_h5(out, [a, b])


def test_mask_slot_pixel_ambiguity(tmp_path):
    """OCR#2: a 2-D mask is rejected when (T,N) == (H,W); per-slot else works."""
    # unambiguous: T,N != H,W -> per-slot
    H, W = 4, 8
    rgb = np.zeros((2, 2, H, W, 3), np.uint8)
    slot = np.array([[True, False], [True, True]])
    out = str(tmp_path / "slot.h5")
    write_scene_h5(out, rgb, mask=slot)
    ok, _ = verify_scene_h5(out, rgb, mask=slot)
    assert ok
    # ambiguous: T == H and N == W (T=4, N=8 == H,W) -> must raise
    rgb2 = np.zeros((4, 8, 4, 8, 3), np.uint8)
    amb = np.ones((4, 8), bool)
    with pytest.raises(ValueError, match=r"ambiguous"):
        write_scene_h5(str(tmp_path / "amb.h5"), rgb2, mask=amb)
    # explicit (T,N,H,W) pixel mask still works in the same scene
    pix = np.ones((4, 8, 4, 8), bool)
    pix[0, 0, 0, 0] = False
    write_scene_h5(str(tmp_path / "pix.h5"), rgb2, mask=pix)
    assert verify_scene_h5(str(tmp_path / "pix.h5"), rgb2, mask=pix)[0]


def test_depth_smaller_than_rgb(tmp_path):
    """depth/depth_conf may be stored at (h,w) < (H,W) if the aspect ratio
    matches; depth_conf must match depth's (h,w); reads resize back to (H,W)."""
    T, N, H, W = 1, 1, 6, 8          # 6/8 ratio
    h, w = 3, 4                      # 3/4 — same ratio, half resolution
    rgb = np.zeros((T, N, H, W, 3), np.uint8)
    depth_mm = np.arange(h * w, dtype=np.uint16).reshape(h, w)   # stored SMALL
    depth_mm[0, 0] = 0              # an invalid sentinel to check survives resize
    conf = np.ones((T, N, h, w), np.float32)
    out = str(tmp_path / "sd.h5")
    write_scene_h5(out, rgb, depth=depth_mm, depth_conf=conf)
    assert verify_scene_h5(out, rgb, depth=depth_mm, depth_conf=conf)[0]
    import h5py
    with h5py.File(out, "r") as hs:
        assert hs["depth"].shape == (T, N, h, w)       # stored small
        assert hs["depth_conf"].shape == (T, N, h, w)
    with H5Scene(out) as s:
        d = s.depth(0, 0)                               # resized back to (H, W)
        assert d.shape == (H, W) and d.dtype == np.uint16
        assert d[0, 0] == 0                            # sentinel (0) survives nearest resize
        assert d[H - 1, W - 1] == h * w - 1           # nearest: bottom-right maps to (h-1,w-1)
        c = s.depth_conf(0, 0)
        assert c.shape == (H, W) and c.all()
        m = s.depth_mask(0, 0)
        assert m.shape == (H, W) and not m[0, 0]      # the sentinel maps to (0,0)
    # depth_conf at a DIFFERENT (h,w) than the depth is rejected
    out = str(tmp_path / "sd_bad.h5")
    with H5Scene(out, T, N, H, W) as s:
        s.set_depth(depth_mm)
        with pytest.raises(ValueError, match=r"depth_conf"):
            s.set_depth_conf(np.ones((1, 1, 2, 2), np.float32))
    # wrong aspect ratio (h,w) is rejected
    out = str(tmp_path / "sd_ratio.h5")
    with pytest.raises(ValueError, match=r"aspect ratio"):
        write_scene_h5(out, rgb, depth=np.ones((T, N, 2, 4), np.uint16))


def test_meta_keys(tmp_path):
    """/meta dataset-side keys: upsert/remove, frame_indices (T,) arange-omission,
    tags sub-group merge, is_metric -> attrs (legacy /meta/is_metric read fallback)."""
    T, N, H, W = 3, 1, 4, 8
    rgb = np.zeros((T, N, H, W, 3), np.uint8)
    out = str(tmp_path / "mk.h5")
    s = H5Scene(out, T, N, H, W)
    try:
        s.set_rgb(rgb)   # an openable H5Scene needs rgb keys; meta test below
        # 7scenes/fire_seq-01/0 -> 4 location keys (see docs/meta_key_decisions.md)
        s.set_meta(dataset="7scenes", scene="fire", seq="seq-01", suffix="0",
                   split="train", frame_indices=np.arange(T),
                   tags={"data_type": "real"})
        assert s.dataset == "7scenes" and s.scene == "fire" and s.seq == "seq-01"
        assert s.suffix == "0"
        assert s.split == "train"
        assert s.frame_indices is None          # == arange(T) -> not stored (dense)
        assert s.tags == {"data_type": "real"}
        # dense arange indices are omitted; a real sparse set is stored
        s.set_meta(frame_indices=[112, 125, 438])
        assert np.array_equal(s.frame_indices, np.array([112, 125, 438]))
        # tags merge: replace existing, add new; upsert + remove of str keys
        s.set_meta(tags={"data_type": "synthetic", "environment": "indoor"}, split=None)
        assert s.tags == {"data_type": "synthetic", "environment": "indoor"}
        assert s.split is None
        s.set_meta(dataset=None, scene=None, seq=None, suffix=None)
        assert s.dataset is None and s.scene is None and s.seq is None and s.suffix is None
    finally:
        s.close()
    # is_metric: attrs on write, /meta fallback on legacy files
    import h5py
    with h5py.File(out, "r+") as h:
        del h.attrs["is_metric"]
        h["meta"].create_dataset("is_metric", data=np.bool_(True))   # legacy layout
    with H5Scene(out) as s2:
        assert s2.is_metric is True                     # read via the /meta fallback
        assert not s2.has("dataset") and s2.has("frame_indices")


def test_open_missing_path_no_orphan(tmp_path):
    """OCR#3: H5Scene(missing) raises FileNotFoundError and leaves no orphan file."""
    p = str(tmp_path / "nope.h5")
    with pytest.raises(FileNotFoundError):
        H5Scene(p)
    import os
    assert not os.path.exists(p)


def test_open_not_h5scene_closes_handle(tmp_path):
    """OCR#3: a non-H5Scene file raises ValueError; handle closed, no crash on reuse."""
    import h5py
    p = str(tmp_path / "plain.h5")
    with h5py.File(p, "w") as h:
        h.create_dataset("x", data=[1, 2, 3])
    with pytest.raises(ValueError, match="not an H5Scene"):
        H5Scene(p)
    # the plain file is still openable and intact (no half-written H5Scene attrs)
    with h5py.File(p, "r") as h:
        assert "TNHW" not in h.attrs and "x" in h


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
    assert s.is_metric
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
        s.set_meta(dataset="7scenes", scene="chess", meta_json='{"a": 2}')
        assert s.dataset == "7scenes" and s.scene == "chess" and s.meta_json == '{"a": 2}'
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
        assert s2.dataset == "7scenes" and s2.scene == "chess" and s2.meta_json == '{"a": 2}'

    # overwrite via create-mode reuse; set_* replaces existing sections
    with H5Scene(out, T, N, H, W) as s3:
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
    test_depth_mask(Path(tempfile.mkdtemp()))
    test_rgb_path_input(Path(tempfile.mkdtemp()))
    test_rgb_jpg_quality_policy(Path(tempfile.mkdtemp()))
    test_rgb_mismatched_resolution(Path(tempfile.mkdtemp()))
    test_mask_slot_pixel_ambiguity(Path(tempfile.mkdtemp()))
    test_depth_smaller_than_rgb(Path(tempfile.mkdtemp()))
    test_meta_keys(Path(tempfile.mkdtemp()))
    test_open_missing_path_no_orphan(Path(tempfile.mkdtemp()))
    test_open_not_h5scene_closes_handle(Path(tempfile.mkdtemp()))
    test_h5scene_file_handle(Path(tempfile.mkdtemp()))
    test_h5scene_set_validation(Path(tempfile.mkdtemp()))
    test_h5scene_absent_sections(Path(tempfile.mkdtemp()))
    print("OK")
