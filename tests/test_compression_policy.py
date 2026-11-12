"""Regression tests pinning the per-section compression decisions (2026-11-12).

Every filter choice below was measured on real probe data (see tests/probes/README.md
and the docstrings in h5scene/__init__.py); these tests keep the choices from
silently drifting when the writer changes. Rerun: pixi run -e test test.
"""

from pathlib import Path
import numpy as np
import h5py
import pytest

from h5scene import H5Cam  # noqa: E402

PROBES = Path(__file__).resolve().parent / "probes"
DEPTH_PROBE = PROBES / "adt_work_skeleton_seq137" / "frame6_depth_mm.npy"
RGB_PROBE = PROBES / "_i025_probe" / "frame6_near_crop.jpg"

H, W = 64, 128


def _full_scene(tmp_path):
    """Scene with every section populated, 2x2 frames of HxW."""
    rng = np.random.default_rng(0)
    out = str(tmp_path / "policy.h5")
    s = H5Cam(out, 2, H, W)
    s.set_rgb(rng.integers(0, 256, (2, H, W, 3)).astype(np.uint8))
    s.set_depth(rng.integers(100, 3000, (2, H, W)).astype(np.uint16))
    m = np.ones((2, H, W), bool)
    m[:, -H // 4:, :] = False
    s.set_mask(m)
    s.set_depth_conf(np.where(m, 1.0, 0.0).astype(np.float32))
    s.set_K(np.array([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]], np.float32))
    s.set_pose(np.eye(4, dtype=np.float32))
    return s, out


# ---- the decision table: section -> (compression, shuffle) ------------------

def test_depth_filters_gzip_shuffle(tmp_path):
    """uint16 mm values stay under one high byte -> shuffle gives gzip long
    runs: measured 3200->1025 KB on a 2x2 640x640 probe (vs 1283 KB plain gzip)."""
    s, out = _full_scene(tmp_path)
    s.close()
    import h5py
    with h5py.File(out) as h:
        d = h["depth"]
        assert d.compression == "gzip" and d.compression_opts == 4
        assert d.shuffle is True
        assert d.dtype == np.uint16


def test_mask_filters_gzip_no_shuffle(tmp_path):
    """bool = 1 byte/pixel: the shuffle byte-permute filter has nothing to
    split (measured identical size on/off) and runs of 0/1 are gzip's forte
    (1602 -> 8.6 KB on a mostly-valid 2x2 640x640 probe)."""
    s, out = _full_scene(tmp_path)
    s.close()
    import h5py
    with h5py.File(out) as h:
        m = h["mask"]
        assert m.compression == "gzip" and m.compression_opts == 4
        assert m.shuffle is False
        assert m.dtype == bool


def test_depth_conf_filters_gzip_no_shuffle(tmp_path):
    """fp16 0-1, most values 1.0: shuffle measured 3% LARGER than plain gzip
    (90.8 vs 93.6 KB on a 2x2 640x640 probe)."""
    s, out = _full_scene(tmp_path)
    s.close()
    import h5py
    with h5py.File(out) as h:
        c = h["depth_conf"]
        assert c.compression == "gzip" and c.compression_opts == 4
        assert c.shuffle is False
        assert c.dtype == np.float16


def test_K_T_stay_raw(tmp_path):
    """Small arrays: at typical sequence lengths gzip makes them BIGGER
    (h5 overhead + block header ~2 KB; measured K 3200->5101 B, T 4096->5402 B
    at 16 frames x 2 cams; break-even ~106 / ~64 frames)."""
    s, out = _full_scene(tmp_path)
    s.close()
    import h5py
    with h5py.File(out) as h:
        for name in ("K", "T"):
            assert h[name].compression is None, f"{name} must stay uncompressed"
            assert h[name].shuffle is False


# ---- size regression on real probe data -------------------------------------

@pytest.mark.skipif(not DEPTH_PROBE.exists(), reason="probe depth missing")
def test_depth_size_beats_raw_and_png_budget(tmp_path):
    """Real 640x640 depth: stored ~24% of raw (195 vs 800 KB) — gzip4+shuffle.
    Also beats the PNG16 encoding of the same array (290.6 KB, 2026-11-12)."""
    d = np.load(DEPTH_PROBE)
    H_, W_ = d.shape
    out = str(tmp_path / "depth_only.h5")
    s = H5Cam(out, 1, H_, W_)
    s.set_depth(d)
    s.close()
    size = Path(out).stat().st_size
    raw = d.nbytes
    assert size < 0.5 * raw, f"depth no longer compresses: {size} B vs {raw} B raw"
    assert size < 300 * 1024, f"depth grew past the ~290 KB PNG16 budget: {size} B"


@pytest.mark.skipif(not RGB_PROBE.exists(), reason="probe rgb missing")
def test_rgb_stored_jpg95_smaller_than_png_and_lossy(tmp_path):
    """The near-crop encodes to ~33 KB as jpg q95 vs 112 KB png (2026-11-12);
    and the encoding is measurably lossy (q100 is NOT lossless either)."""
    import cv2
    img = cv2.imdecode(np.frombuffer(RGB_PROBE.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    h_, w_ = img.shape[:2]
    out = str(tmp_path / "rgb_only.h5")
    s = H5Cam(out, 1, h_, w_)
    s.set_rgb([img])
    s.close()
    stored = h5py.File(out, "r")["rgb"]["0"][:]   # raw stored bytes
    png = cv2.imencode(".png", img)[1]
    assert len(stored) < 0.5 * len(png), (
        f"rgb no longer smaller than png: {len(stored)} vs {len(png)} B — "
        "is the writer back to lossless encoding?")
    dec = cv2.imdecode(np.frombuffer(stored, np.uint8), cv2.IMREAD_COLOR)
    assert np.abs(dec.astype(int) - img.astype(int)).max() > 0, \
        "rgb encoding is lossless now (expected jpg q95)"
