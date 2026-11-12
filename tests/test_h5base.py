"""Tests for H5Base (the file handle): lifecycle, the generic attr/meta
accessors (registry-checked, type-checked), write_section."""

from pathlib import Path
import numpy as np
import pytest
import h5py

from h5scene import H5Base, H5Cam  # noqa: E402


def test_open_missing_raises_no_orphan(tmp_path):
    p = str(tmp_path / "missing.h5")
    with pytest.raises(FileNotFoundError):
        H5Base(p)
    assert not Path(p).exists()


def test_create_open_roundtrip(tmp_path):
    p = str(tmp_path / "base.h5")
    with H5Base(p, create=True) as b:
        assert b.h is not None and b.path == p
        assert "meta" in b.h                     # the base creates the /meta group
    with H5Base(p) as b:
        assert b.h is not None and b.h.attrs is not None
        assert isinstance(b.h, h5py.File) and b.h.mode == "r"


def test_write_section(tmp_path):
    with H5Base(str(tmp_path / "base.h5"), create=True) as b:
        b.write_section("x", np.arange(4, dtype=np.float32))
        b.write_section("x", np.arange(8, dtype=np.float32))   # replaces
    with h5py.File(str(tmp_path / "base.h5"), "r") as h:
        assert h["x"].shape == (8,)


def test_subclass_registries_are_checked(tmp_path):
    p = str(tmp_path / "cam.h5")
    with H5Cam.create(p, 2, 4, 4) as cam:
        cam.set_attr("THW", np.array([2, 4, 4], np.int64))
        cam.set_attr("is_metric", True)
        H5Base.set_meta(cam, "scene", "s")            # the base single-key accessor
        H5Base.set_meta(cam, "tags", {"data_type": "syn"})
        H5Base.set_meta(cam, "src_frame_idxs", np.array([3, 7]))
        assert bool(cam.attr("is_metric")) is True
        assert cam.meta("scene") == "s"
        assert cam.meta("tags") == {"data_type": "syn"}
        assert list(cam.meta("src_frame_idxs")) == [3, 7]
        # unknown keys (both accessors are registry-checked)
        with pytest.raises(ValueError, match="unknown attr"):
            cam.set_attr("bogus", 1)
        with pytest.raises(ValueError, match="unknown meta key"):
            H5Base.set_meta(cam, "bogus", "x")
        # wrong types (isinstance-checked against the declared type)
        with pytest.raises(TypeError):
            cam.set_attr("is_metric", "yes")
        with pytest.raises(TypeError):
            H5Base.set_meta(cam, "scene", 42)
        # single-key None removes
        H5Base.set_meta(cam, "scene", None)
        assert cam.meta("scene") is None
        # the multi-key convenience sets only the keys given
        assert cam.set_meta(dataset="d", seq="q") is cam
        assert cam.dataset == "d" and cam.seq == "q" and cam.scene is None


def test_rig_registries_have_no_seq(tmp_path):
    from h5scene import H5CamRig
    p = str(tmp_path / "rig.h5")
    # two identical cams (same dataset/scene), minimal rgb
    import cv2
    cams = []
    for i in (0, 1):
        cp = str(tmp_path / f"c{i}.h5")
        with H5Cam.create(cp, 2, 8, 8) as c:
            c.set_rgb([cv2.imencode(".jpg", np.full((8, 8, 3), i + 1, np.uint8))[1].tobytes()] * 2)
            c.set_meta(dataset="d", scene="s")
        cams.append(cp)
    with H5CamRig.build(p, cams, suffix="v") as rig:
        assert "seq" not in H5CamRig.META_KEYS
        assert "seq" not in H5CamRig.ATTR_KEYS
        assert rig.dataset == "d" and rig.scene == "s" and rig.suffix == "v"
        assert "seq" not in type(rig).__dict__   # no seq property at all
        assert not hasattr(rig, "seq")
        assert len(rig.cam_files) == 2 and list(rig.cam_hw) == [(8, 8), (8, 8)]


def test_meta_registry_is_all_metasp(tmp_path):
    """#48: every META_KEYS entry is a MetaSpec (no plain type left) and the
    tags value check (str keys, str values) lives in MetaSpec.check."""
    from h5scene import H5Cam
    from h5scene.h5base import MetaSpec
    for spec in H5Cam.META_KEYS.values():
        assert isinstance(spec, MetaSpec), spec
    # a non-str tags value / key is rejected by the spec, not a legacy branch
    class C(H5Cam):
        pass
    with H5Cam.create(str(tmp_path / "c.h5"), 2, 4, 4) as cam:
        with pytest.raises(TypeError, match="must be a str"):
            cam.set_meta(tags={"a": 5})
        cam.set_meta(tags={"a": "x"})
        assert cam.meta("tags") == {"a": "x"}
