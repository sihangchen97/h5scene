"""H5Geo: static world-frame geometry — a point cloud or a mesh, no time axis.

Houdini element levels (a VERTEX is a face corner, never a position):

    <scene>.geo.h5
      attrs  n_points int64 M (pinned at create), is_metric bool,
             prim_size int64 3/4 fixed or 0 mixed (present iff topology)
      /points/<attr>    rows M:  P (M,3) f32 required, Cd (M,3) u8,
                        N (M,3) f32, … (the full table: codecs/sections.py
                        GEO_ATTRS)
      /vertices/<attr>  rows V = len(ptnum): ptnum (V,) i32 required iff
                        topology, N/uv/Cd …
      /prims/<attr>     rows F:  vertexcount (F,) i32 only when prim_size 0,
                        N/Cd/class/material/id …
      /meta/…           H5Cam keys minus seq / src_frame_idxs / src_frame_ids /
                        timestamps (a static scene has no per-frame ids)
      /detail/…         name tables + textures (Phase 2 / deferred)

V and F are derived (never stored): V = len(ptnum), F = V // prim_size for
fixed-size prims or len(vertexcount) for mixed polygons. No ptnum = a point
cloud (V = F = 0). The one length rule: /points/* axis 0 = M, /vertices/*
= V, /prims/* = F — checked at open (shapes/dtypes only, O(1)) and at write
time; VALUE-range checks (ptnum range, vertexcount >= 3 / sum = V, ids >= -1)
run at write and read time (D5), not at open.

Phase 1 (this file): P / Cd / N on points + triangle topology. Phase 2 reaches
the other levels and the /detail tables through the same generic accessors.
"""
import h5py
import numpy as np

from .codecs import CUSTOM_DTYPES, GEO_ATTRS, geo_chunks, is_metric as _is_metric
from .h5base import H5Base, MetaSpec
from .h5cam import H5Cam

_I32_MIN = np.iinfo(np.int32).min
_I32_MAX = np.iinfo(np.int32).max
_ID_NAMES = ("class", "id", "material")


class H5Geo(H5Base):
    """Static world-frame geometry: attrs.n_points = M (pinned at create),
    /points/P (M,3) f32 required, /points/{Cd,N} optional, triangle topology
    as attrs.prim_size=3 + /vertices/ptnum (V,) i32. create(path, P) pins M
    (a create failure removes the half-made file); open requires n_points /
    is_metric / /points/P and, when ptnum is present, prim_size with a
    consistent V and F. A point cloud has no ptnum (V = F = 0)."""

    ATTR_KEYS = {"file_type": str,      # always "h5geo" (pinned at create)
                  "n_points": int,      # M, pinned at create
                  "is_metric": bool,    # default False (D3)
                  "prim_size": int}     # 3 / 4 = fixed, 0 = mixed (iff topology)
    META_KEYS = {k: v for k, v in H5Cam.META_KEYS.items()
                  if k not in ("seq", "src_frame_idxs", "src_frame_ids",
                               "timestamps")}

    LEVELS = ("points", "vertices", "prims")

    def __init__(self, path, mode="r"):
        """Open an existing H5Geo file ("r" or "r+"). Construction goes
        through H5Geo.create — not this."""
        super().__init__(path, mode=mode)
        try:
            ft = self._h.attrs.get("file_type")
            if ft is None or str(ft) != "h5geo":
                raise ValueError(f"{self.path}: not an H5Geo file "
                                 f"(attrs.file_type = {ft!r})")
            m = self.attr("n_points")
            if m is None or "is_metric" not in self._h.attrs:
                raise ValueError(f"{self.path}: not an H5Geo file "
                                 "(no attrs.n_points / is_metric)")
            self.M = int(m)
            self._check_geometry()
        except Exception:                       # corrupt file, bad counts, …
            self.close()
            raise

    # ---- construction -------------------------------------------------------
    @classmethod
    def create(cls, path, P, *, is_metric=False):
        """Create a new file (overwriting): pins n_points = len(P) and writes
        /points/P. P: (M,3) float (f64 stored as f32; NaN/inf rejected);
        M >= 1. A constructor failure removes the half-made file."""
        P = np.asarray(P)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError(f"create needs (M, 3) P, got {P.shape}")
        if P.dtype not in (np.float32, np.float64):
            raise TypeError(f"create needs float P (f32/f64), got {P.dtype.name}")
        if P.shape[0] < 1:
            raise ValueError("create needs M >= 1")
        if not np.all(np.isfinite(P)):
            raise ValueError("P has NaN/inf")
        geo = object.__new__(cls)               # bypass the open-only __init__
        H5Base.__init__(geo, path, create=True)
        geo._V, geo._F = 0, 0                   # no topology yet
        try:
            geo.set_attr("file_type", "h5geo")
            geo.M = P.shape[0]
            geo.set_attr("n_points", geo.M)
            geo.set_attr("is_metric", bool(is_metric))
            for g in cls.LEVELS:
                geo._h.create_group(g)
            geo.set_P(P)
        except Exception:
            geo._abort_create()
            raise
        return geo

    @classmethod
    def open(cls, path, mode="r"):
        """Open an existing file ("r" or "r+")."""
        return cls(path, mode=mode)

    # ---- open-time checks (O(1) in data size: shapes / dtypes only) --------
    def _check_geometry(self):
        h = self._h
        pts = h.get("points")
        if pts is None or "P" not in pts:
            raise ValueError("missing /points/P")
        p = pts["P"]
        if p.dtype != np.float32 or p.shape != (self.M, 3):
            raise ValueError(f"/points/P shape {p.shape} / dtype {p.dtype.name} "
                             f"!= (M={self.M}, 3) f32")
        top = "ptnum" in (h.get("vertices") or {})
        ps = self.attr("prim_size")
        if (ps is not None) != top:
            raise ValueError("prim_size must be present exactly when ptnum is")
        if top:
            if ps not in (0, 3, 4):
                raise ValueError(f"prim_size {ps} not in (0, 3, 4)")
            V = h["vertices/ptnum"].shape[0]
            if ps in (3, 4) and V % ps:
                raise ValueError(f"V = {V} not divisible by prim_size {ps}")
            vc_present = "vertexcount" in h["prims"]
            if vc_present != (ps == 0):
                raise ValueError("vertexcount must be present exactly when "
                                 "prim_size == 0")
            self._V, self._F = V, (V // ps if ps in (3, 4) else h["prims/vertexcount"].shape[0])
        else:
            self._V, self._F = 0, 0
        for lvl in self.LEVELS:
            grp = h.get(lvl)
            if grp is None:
                continue
            count = {"points": self.M, "vertices": self._V, "prims": self._F}[lvl]
            for name in grp:
                spec = GEO_ATTRS.get(f"{lvl}/{name}")
                d = grp[name]
                if not isinstance(d, h5py.Dataset):
                    raise ValueError(f"{lvl}/{name} is not a dataset (a nested "
                                     "group corrupts the file — attribute names "
                                     "must not contain '/')")
                if spec is not None:
                    if d.dtype != spec.dtype:
                        raise ValueError(f"{lvl}/{name} dtype {d.dtype.name} "
                                         f"!= {np.dtype(spec.dtype).name}")
                    if d.ndim != (1 if spec.cols == 1 else 2) or \
                            (spec.cols != 1 and d.shape[1] != spec.cols):
                        raise ValueError(f"{lvl}/{name} shape {d.shape} != "
                                         f"(N{', ' + str(spec.cols) if spec.cols != 1 else ''})")
                elif d.dtype not in CUSTOM_DTYPES and d.dtype != np.bool_:
                    raise ValueError(f"{lvl}/{name} custom dtype {d.dtype.name} "
                                     f"not in {tuple(np.dtype(t).name for t in CUSTOM_DTYPES)}")
                elif d.ndim not in (1, 2):
                    raise ValueError(f"{lvl}/{name} custom shape {d.shape} not (N[, c])")
                if d.shape[0] != count:
                    raise ValueError(f"{lvl}/{name} axis 0 = {d.shape[0]} != "
                                     f"{lvl} count {count}")

    # ---- counts (only n_points + prim_size are stored) ----------------------
    @property
    def n_points(self):
        return self.M

    @property
    def n_vertices(self):
        return self._V

    @property
    def n_prims(self):
        return self._F

    @property
    def prim_size(self):
        return self.attr("prim_size")

    @property
    def has_topology(self):
        return self._V != 0

    @property
    def is_triangles(self):
        if not self.has_topology:
            return False
        ps = self.prim_size
        if ps == 3:
            return True
        if ps == 0:
            return bool((self._h["prims/vertexcount"][:] == 3).all())
        return False

    def __len__(self):
        return self.M

    @property
    def is_metric(self):
        return _is_metric(self._h)

    def has(self, name):
        """Presence of a GEO_ATTRS key ('points/P'), a bare point-level name
        ('P'), a bare vertex/prim name ('ptnum'), or /meta/<name>."""
        if name in GEO_ATTRS or "/" in name:
            return name in self._h
        for lvl in self.LEVELS:
            grp = self._h.get(lvl)
            if grp is not None and name in grp:
                return True
        return name in self.META_KEYS and self.meta(name) is not None

    def __repr__(self):
        out = [f"<{type(self).__name__} {self.path} M={self.M}"]
        if self.has_topology:
            out.append(f" V={self._V} F={self._F} prim_size={self.prim_size}")
        for lvl in self.LEVELS:
            a = sorted(self.point_attrs if lvl == "points"
                       else self._level_attrs(lvl))
            if a:
                out.append(f" {lvl}={a}")
        return " ".join(out) + ">"

    # ---- /meta (H5Cam keys minus seq / src_frame_idxs / src_frame_ids /
    #      timestamps — a static scene has no per-frame ids) ------------------
    @property
    def dataset(self):
        return self.meta("dataset")

    @property
    def scene(self):
        return self.meta("scene")

    @property
    def suffix(self):
        return self.meta("suffix")

    @property
    def split(self):
        return self.meta("split")

    @property
    def tags(self):
        return self.meta("tags")

    # ---- topology (written as one unit; value-checked at write AND read) ----
    def set_triangles(self, tri):
        """(F,3) int/uint triangle indices, 0 <= i < M -> attrs.prim_size = 3
        and /vertices/ptnum = tri.ravel() (V = 3F). Indices are range-checked
        BEFORE anything is deleted; an existing topology of a different size
        (or with vertex/prim attributes) must be del_topology()'d first."""
        a = np.asarray(tri)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"set_triangles needs (F, 3), got {a.shape}")
        if a.shape[0] < 1:
            raise ValueError("set_triangles needs F >= 1 (an empty mesh is a "
                             "point cloud — just don't write a topology)")
        if a.dtype.kind not in "iu":
            raise TypeError(f"set_triangles needs int/uint indices, "
                             f"got {a.dtype.name}")
        if a.min() < 0 or a.max() >= self.M:
            raise ValueError(f"triangle index out of range 0..{self.M - 1} "
                             f"(min {a.min()}, max {a.max()})")
        if a.max().item() > _I32_MAX or a.min().item() < _I32_MIN:
            raise ValueError(f"triangle index out of int32 range")
        h = self._h
        V = 3 * a.shape[0]
        if self.has_topology:
            if self.prim_size != 3 or self._V != V:
                raise ValueError("changing the topology: del_topology() first")
            for lvl in ("vertices", "prims"):
                for name in h[lvl]:
                    if not (lvl, name) == ("vertices", "ptnum"):
                        raise ValueError(f"changing the topology while "
                                         f"{lvl}/{name} exists: del_topology() first")
        # everything checked — write the unit (prim_size, then ptnum)
        if "ptnum" in h["vertices"]:
            del h["vertices/ptnum"]
        h["vertices"].create_dataset(
            "ptnum", data=np.ascontiguousarray(a.ravel(), np.int32),
            chunks=geo_chunks(V, 1, GEO_ATTRS["vertices/ptnum"].chunk_rows),
            **self._filters("vertices/ptnum"))
        self.set_attr("prim_size", 3)
        self._V, self._F = V, a.shape[0]
        return self

    def triangles(self):
        """(F,3) i32 triangle indices (ptnum un-flattened). prim_size 3, or
        0 with EVERY vertexcount == 3; anything else raises. The ptnum range
        is checked on read (D5)."""
        if not self.has_topology:
            raise KeyError("no topology (point cloud)")
        ps = self.prim_size
        if ps == 0:
            vc = self._h["prims/vertexcount"][:]
            if (vc < 3).any() or vc.sum() != self._V:
                raise ValueError("vertexcount invalid (each >= 3, sum = V)")
            if not (vc == 3).all():
                raise ValueError("mixed vertexcounts: no single (F,3) triangle array")
        elif ps != 3:
            raise ValueError(f"prim_size {ps}, not triangles")
        return self._ptnum().reshape(-1, 3)

    def del_topology(self):
        """Remove the whole topology unit: /vertices/ptnum, /prims/vertexcount,
        attrs.prim_size, and every /vertices/* and /prims/* dataset."""
        h = self._h
        for key in ("vertices/ptnum", "prims/vertexcount"):
            if key in h:
                del h[key]
        if "prim_size" in h.attrs:
            del h.attrs["prim_size"]
        for lvl in ("vertices", "prims"):
            for name in list(h.get(lvl) or {}):
                del h[lvl][name]
        self._V, self._F = 0, 0
        return self

    # ---- generic per-level attribute accessors (Houdini names) --------------
    def point_attr(self, name):
        return self._get("points", name)

    def set_point_attr(self, name, a):
        return self._set("points", name, a)

    def del_point_attr(self, name):
        return self._del("points", name)

    def vertex_attr(self, name):
        return self._get("vertices", name)

    def set_vertex_attr(self, name, a):
        return self._set("vertices", name, a)

    def del_vertex_attr(self, name):
        return self._del("vertices", name)

    def prim_attr(self, name):
        return self._get("prims", name)

    def set_prim_attr(self, name, a):
        return self._set("prims", name, a)

    def del_prim_attr(self, name):
        return self._del("prims", name)

    @property
    def point_attrs(self):
        return self._level_attrs("points")

    @property
    def vertex_attrs(self):
        return self._level_attrs("vertices")

    @property
    def prim_attrs(self):
        return self._level_attrs("prims")

    # ---- Phase 1 shortcuts (point level) ------------------------------------
    def P(self):
        """(M,3) f32 world positions (metres when is_metric)."""
        return self._get("points", "P")

    def set_P(self, P):
        """(M,3) float32/64 positions — a replacement must keep M."""
        return self._set("points", "P", P)

    def Cd(self):
        """(M,3) u8 point colours (open3d 0-1 floats: np.round(c*255))."""
        return self._get("points", "Cd")

    def set_Cd(self, Cd):
        """(M,3) uint8 ONLY (floats rejected — D4)."""
        return self._set("points", "Cd", Cd)

    def N(self):
        """(M,3) f32 point normals."""
        return self._get("points", "N")

    def set_N(self, N):
        """(M,3) float32/64 point normals."""
        return self._set("points", "N", N)

    # ---- derived helpers (never stored) -------------------------------------
    def bbox(self):
        """(min (3,) f32, max (3,) f32) of P."""
        p = self.P()
        return p.min(axis=0), p.max(axis=0)

    # ---- internals -----------------------------------------------------------
    def _level_attrs(self, lvl):
        grp = self._h.get(lvl)
        return set(grp) if grp is not None else set()

    def _level_count(self, lvl):
        return {"points": self.M, "vertices": self._V, "prims": self._F}[lvl]

    def _filters(self, key, out=None):
        """Standard keys: the table row; custom (D10, `out` given): the filter
        rule by dtype (multi-byte gzip4+shuffle, 1-byte plain gzip4)."""
        spec = GEO_ATTRS.get(key)
        if spec is not None:
            comp, opts, shfl = spec.compression, spec.compression_opts, spec.shuffle
        else:
            comp, opts = "gzip", 4
            shfl = np.dtype(out.dtype).itemsize > 1
        return {"compression": comp, "compression_opts": opts, "shuffle": shfl}

    def _coerce(self, key, spec, a):
        """Cast to the stored dtype (the only legal casts; anything else raises)
        + the value checks. Returns the array to store."""
        name = key.split("/", 1)[1]
        if spec is not None:
            want = spec.dtype
            if a.dtype == want:
                out = a
            elif want == np.float32 and a.dtype == np.float64:
                out = a.astype(np.float32)
            elif want == np.int32 and a.dtype.kind == "i":
                if a.size and (a.max() > _I32_MAX or a.min() < _I32_MIN):
                    raise ValueError(f"{key} value out of int32 range")
                out = a.astype(np.int32)
            else:
                raise TypeError(f"{key} wants {np.dtype(want).name}, "
                                f"got {a.dtype.name}")
        else:                                              # custom (D10)
            if a.dtype not in CUSTOM_DTYPES and a.dtype != np.bool_:
                raise TypeError(f"custom {key} dtype {a.dtype.name} not in "
                                f"{tuple(np.dtype(t).name for t in CUSTOM_DTYPES)}")
            out = a
        if np.issubdtype(out.dtype, np.floating) and not np.all(np.isfinite(out)):
            raise ValueError(f"{key} has NaN/inf")
        if name in _ID_NAMES and out.size and out.min() < -1:
            raise ValueError(f"{key} ids must be >= -1 (got min {out.min()})")
        return np.ascontiguousarray(out)

    def _set(self, lvl, name, a):
        if "/" in name:
            raise ValueError(f"attribute name {name!r} must not contain '/' "
                             f"(it would create a nested group)")
        key = f"{lvl}/{name}"
        spec = GEO_ATTRS.get(key)
        if lvl != "points" and not self.has_topology:
            raise ValueError(f"{key} needs topology (ptnum) — set_triangles() first")
        a = np.asarray(a)
        count = self._level_count(lvl)
        if a.ndim < 1 or a.shape[0] != count:
            raise ValueError(f"{key} axis 0 = {a.shape[0] if a.ndim else 0} "
                             f"!= {lvl} count {count}")
        if spec is not None:
            if spec.cols == 1:
                if a.ndim != 1:
                    raise ValueError(f"{key} must be 1-D (N,), got {a.shape}")
            elif a.ndim != 2 or a.shape[1] != spec.cols:
                raise ValueError(f"{key} must be (N, {spec.cols}), got {a.shape}")
        elif a.ndim not in (1, 2):
            raise ValueError(f"custom {key} must be (N[, c]), got {a.shape}")
        out = self._coerce(key, spec, a)        # validated before anything is deleted
        cols = spec.cols if spec is not None else (1 if out.ndim == 1 else out.shape[1])
        rows = spec.chunk_rows if spec is not None else 65_536
        grp = self._h[lvl]
        if key in self._h:
            del grp[name]
        grp.create_dataset(name, data=out,
                           chunks=geo_chunks(count, cols, rows),
                           **self._filters(key, out))
        return self

    def _get(self, lvl, name):
        key = f"{lvl}/{name}"
        if key not in self._h:
            raise KeyError(f"no {key}")
        a = self._h[key][:]
        if (lvl, name) == ("vertices", "ptnum"):
            if a.size and (a.min() < 0 or a.max() >= self.M):
                raise ValueError(f"ptnum out of range 0..{self.M - 1} (read check, D5)")
        return a

    def _del(self, lvl, name):
        key = f"{lvl}/{name}"
        if key in self._h:
            del self._h[key]
        return self

    def _ptnum(self):
        return self._get("vertices", "ptnum")
