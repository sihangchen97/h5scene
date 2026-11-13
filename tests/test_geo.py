"""H5Geo Phase 1 tests (requirements_gt_geometry/IMPLEMENTATION_SPEC.md §8,
items 1–9). The §5.3 compression tests live in test_compression_policy.py."""
import numpy as np
import h5py
import pytest

from h5scene import H5Geo


def _mesh(M=8, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.random((M, 3), dtype=np.float32) * 10
    Cd = rng.integers(0, 256, (M, 3)).astype(np.uint8)
    N = rng.random((M, 3), dtype=np.float32)
    tri = np.array([[i, i + 1, i + 2] for i in range(M - 2)])
    return P, Cd, N, tri


# ---- 1. triangle mesh round-trip --------------------------------------------

def test_mesh_round_trip(tmp_path):
    P, Cd, N, tri = _mesh()
    f = str(tmp_path / "mesh.geo.h5")
    g = H5Geo.create(f, P)
    g.set_Cd(Cd)
    g.set_N(N)
    g.set_triangles(tri)
    g.close()
    g2 = H5Geo.open(f)
    assert g2.M == 8 and g2.n_vertices == 3 * (len(tri)) and g2.n_prims == len(tri)
    assert g2.prim_size == 3 and g2.is_triangles
    assert np.array_equal(g2.P(), P)
    assert np.array_equal(g2.Cd(), Cd)
    assert np.array_equal(g2.N(), N)
    assert np.array_equal(g2.triangles(), tri)
    assert g2.point_attrs == {"P", "Cd", "N"}


def test_set_triangles_empty_raises_cleanly(tmp_path):
    """A review pass: set_triangles with F = 0 must raise a clear message
    (not numpy's zero-size reduction error)."""
    g = H5Geo.create(str(tmp_path / "pc.geo.h5"), np.zeros((4, 3), np.float32))
    with pytest.raises(ValueError, match="F >= 1"):
        g.set_triangles(np.zeros((0, 3), dtype=np.int32))
    g.close()


# ---- 2. point-cloud round-trip -----------------------------------------------

def test_point_cloud_round_trip(tmp_path):
    P, Cd, N, _ = _mesh()
    f = str(tmp_path / "pc.geo.h5")
    H5Geo.create(f, P).close()
    g2 = H5Geo.open(f)
    assert g2.M == 8 and g2.n_vertices == 0 and g2.n_prims == 0
    assert not g2.has_topology and g2.prim_size is None
    with pytest.raises(KeyError):
        g2.triangles()


# ---- 3. f64 P is stored as f32; float Cd is rejected -------------------------

def test_f64_P_stored_f32(tmp_path):
    P64 = np.random.default_rng(1).random((4, 3), dtype=np.float64)
    g = H5Geo.create(str(tmp_path / "a.geo.h5"), P64)
    assert g.P().dtype == np.float32
    assert np.array_equal(g.P(), P64.astype(np.float32))
    g.close()
    g2 = H5Geo.open(g.path)
    assert g2.h["points/P"].dtype == np.float32


def test_float_Cd_rejected(tmp_path):
    g = H5Geo.create(str(tmp_path / "b.geo.h5"), np.zeros((4, 3), np.float32))
    with pytest.raises(TypeError):
        g.set_Cd(np.ones((4, 3), np.float32))
    # the converter path: 0-1 floats -> round(c*255) u8
    c01 = np.random.default_rng(2).random((4, 3))
    g.set_Cd(np.round(c01 * 255).astype(np.uint8))
    assert g.Cd().dtype == np.uint8


def test_P_replacement_keeps_M(tmp_path):
    g = H5Geo.create(str(tmp_path / "c.geo.h5"), np.zeros((4, 3), np.float32))
    with pytest.raises(ValueError):
        g.set_P(np.zeros((5, 3), np.float32))


# ---- 4. bad triangle indices -------------------------------------------------

def test_set_triangles_rejects_bad_indices(tmp_path):
    g = H5Geo.create(str(tmp_path / "d.geo.h5"), np.zeros((4, 3), np.float32))
    with pytest.raises(ValueError):           # index == M
        g.set_triangles(np.array([[0, 1, 4]], np.int32))
    with pytest.raises(ValueError):           # index -1
        g.set_triangles(np.array([[0, 1, -1]], np.int32))
    with pytest.raises(TypeError):            # float array
        g.set_triangles(np.array([[0, 1, 2]], np.float32))
    with pytest.raises(ValueError):           # not (F,3)
        g.set_triangles(np.array([[0, 1]], np.int32))


def test_bad_ptnum_via_raw_h5py(tmp_path):
    """D5: an out-of-range ptnum written through raw h5py still opens (shape
    checks only); the read-time value check raises."""
    f = str(tmp_path / "e.geo.h5")
    g = H5Geo.create(f, np.zeros((4, 3), np.float32))
    g.set_triangles(np.array([[0, 1, 2]], np.int32))
    g.close()
    with h5py.File(f, "a") as h:              # corrupt ptnum by hand
        h["vertices/ptnum"][0] = 7
    g2 = H5Geo.open(f)                        # open: fine (O(1) checks only)
    with pytest.raises(ValueError):
        g2.triangles()


# ---- 5. missing P / missing root attr: open raises, handle closed ------------

def test_missing_P_raises_closed(tmp_path):
    f = str(tmp_path / "f.geo.h5")
    H5Geo.create(f, np.zeros((4, 3), np.float32)).close()
    with h5py.File(f, "a") as h:
        del h["points/P"]
    with pytest.raises(ValueError):
        H5Geo.open(f)


def test_missing_root_attr_raises_closed(tmp_path):
    f = str(tmp_path / "g.geo.h5")
    H5Geo.create(f, np.zeros((4, 3), np.float32)).close()
    with h5py.File(f, "a") as h:
        del h.attrs["n_points"]
    with pytest.raises(ValueError):
        H5Geo.open(f)


# ---- 6. length rule -----------------------------------------------------------

def test_length_rule_write_rejects_M_plus_1(tmp_path):
    g = H5Geo.create(str(tmp_path / "h.geo.h5"), np.zeros((4, 3), np.float32))
    with pytest.raises(ValueError):
        g.set_Cd(np.zeros((5, 3), np.uint8))
    with pytest.raises(ValueError):
        g.set_N(np.zeros((3, 3), np.float32))
    # the rejected write kept the old data absent (nothing was written)
    assert not g.has("Cd")


def test_length_rule_hand_edited_Cd_rejected_at_open(tmp_path):
    f = str(tmp_path / "i.geo.h5")
    g = H5Geo.create(f, np.zeros((4, 3), np.float32))
    g.set_Cd(np.zeros((4, 3), np.uint8))
    g.close()
    with h5py.File(f, "a") as h:              # hand-edit: axis 0 = 5 != M = 4
        del h["points/Cd"]
        h["points"].create_dataset("Cd", data=np.zeros((5, 3), np.uint8))
    with pytest.raises(ValueError):
        H5Geo.open(f)


# ---- 7. unknown datasets under /points ----------------------------------------

def test_unknown_point_dataset_rejected_at_open(tmp_path):
    f = str(tmp_path / "j.geo.h5")
    H5Geo.create(f, np.zeros((4, 3), np.float32)).close()
    with h5py.File(f, "a") as h:
        h["points"].create_dataset("bogus", data=np.zeros((4,), np.float64))
    with pytest.raises(ValueError):            # float64 not in the custom whitelist
        H5Geo.open(f)


def test_custom_attribute_allowed(tmp_path):
    """D10: a custom name with a whitelisted dtype + the length rule passes."""
    g = H5Geo.create(str(tmp_path / "k.geo.h5"), np.zeros((4, 3), np.float32))
    g.set_point_attr("elevation", np.arange(4, dtype=np.float32))
    assert g.point_attrs == {"P", "elevation"}
    assert np.array_equal(g.point_attr("elevation"), np.arange(4, dtype=np.float32))
    g.del_point_attr("elevation")
    assert g.point_attrs == {"P"}
    with pytest.raises(KeyError):
        g.point_attr("elevation")


# ---- 8. failed operations ------------------------------------------------------

def test_failed_write_keeps_old_data(tmp_path):
    g = H5Geo.create(str(tmp_path / "l.geo.h5"), np.zeros((4, 3), np.float32))
    good = np.full((4, 3), 7, np.uint8)
    g.set_Cd(good)
    with pytest.raises(TypeError):
        g.set_Cd(np.ones((4, 3), np.float32))     # rejected: float
    assert np.array_equal(g.Cd(), good)           # old data intact


def test_failed_create_removes_file(tmp_path):
    f = str(tmp_path / "m.geo.h5")
    with pytest.raises(ValueError):
        H5Geo.create(f, np.zeros((4, 2), np.float32))    # not (M,3)
    assert not __import__("os").path.exists(f)


# ---- 9. meta -------------------------------------------------------------------

def test_meta_keys(tmp_path):
    g = H5Geo.create(str(tmp_path / "n.geo.h5"), np.zeros((4, 3), np.float32))
    assert g.is_metric is False                 # default False (D3)
    g.set_meta(dataset="spatialbench", scene="chess_seq-05", split="train",
               tags={"kind": "gt"})
    assert g.dataset == "spatialbench" and g.scene == "chess_seq-05"
    assert g.split == "train" and g.tags == {"kind": "gt"} and g.suffix is None
    with pytest.raises(ValueError):             # camera-only keys are not ours
        g.set_meta("seq", "s")
    with pytest.raises(ValueError):
        g.set_meta("src_frame_idxs", np.array([0], np.int64))
    # and they round-trip on open
    g.close()
    g2 = H5Geo.open(g.path)
    assert g2.dataset == "spatialbench" and g2.scene == "chess_seq-05"
    assert g2.is_metric is False


def test_is_metric_written(tmp_path):
    g = H5Geo.create(str(tmp_path / "o.geo.h5"), np.zeros((4, 3), np.float32),
                     is_metric=True)
    assert g.is_metric is True
    g.close()
    assert H5Geo.open(g.path).is_metric is True
