"""H5Base: the base per-scene HDF5 FILE HANDLE (no in-memory data): the open
h5py.File, its path, lifecycle (close / context manager), the generic
attr(name) / set_attr(name, value) + meta(name) / set_meta(name, value)
accessors, and the one low-level dataset write (write_section).

Subclasses declare their OWN known-key registries as class attributes — the
base has none:
    ATTR_KEYS = {name: type, …}   # checked by attr() / set_attr()
    META_KEYS = {name: MetaSpec, …}  # checked by meta() / set_meta()
the declared type is checked via isinstance (numpy equivalents of the scalar
types are accepted: np.bool_ for bool, np.integer for int, …); array keys
carry a MetaSpec(type, dtype=int64, length="T") that also checks dtype and
axis-0 length. (H5Cam and H5CamRig each define what their file actually
holds — e.g. the rig has no "seq"; its registry is DERIVED from H5Cam's:
cam keys minus "seq".)

Nothing scene-specific — no shape: the axis convention (T,H,W vs
T,N,Hmax,Wmax) and the section semantics live on the subclasses, which build
their set_*/read methods on the shared helpers in codecs/:

    H5Cam(H5Base)     — single-camera (T, H, W), attrs.THW
    H5CamRig(H5Base)  — multi-camera (T, N, Hmax, Wmax), references only

A subclass that creates a file wraps its post-create setup in try/except and
on failure calls `self._abort_create()` (handle closed, half-made file
removed) so a bad T/H/W never leaves an orphan behind.
"""
import os
import os.path as osp

import h5py
import numpy as np

_STR = h5py.string_dtype(encoding="utf-8")

# numpy equivalents accepted for the scalar declared types
_NUM_EQUIV = {bool: (bool, np.bool_),
              int: (int, np.integer),
              float: (float, np.floating)}


class MetaSpec:
    """A meta key's declared type (+ optional int64 dtype / axis-0 length "T").
    Subclasses derive their registries from it:
        META_KEYS = {k: v for k, v in H5Cam.META_KEYS.items() if k != "seq"}
    """
    __slots__ = ("type", "dtype", "length")

    def __init__(self, type_, dtype=None, length=None):
        self.type = type_
        self.dtype = dtype          # int64 for array keys (checked on write)
        self.length = length        # axis-0 length; "T" = the scene's T

    def check(self, name, value, T=None):
        ok = _NUM_EQUIV.get(self.type, (self.type,))
        if not isinstance(value, ok):
            raise TypeError(f"meta {name!r} expects {self.type.__name__}, "
                            f"got {type(value).__name__}")
        if self.dtype is not None:
            if not isinstance(value, np.ndarray) or value.dtype != self.dtype:
                raise TypeError(f"meta {name!r} expects a {np.dtype(self.dtype).name} "
                                f"array, got {value.dtype}")
        if self.length is not None:
            if not isinstance(value, np.ndarray) or value.ndim != 1:
                raise ValueError(f"meta {name!r} must be a 1-D array")
            L = T if self.length == "T" else self.length
            if L is not None and value.shape[0] != L:
                raise ValueError(f"meta {name!r} must be length {L}, "
                                 f"got {value.shape[0]}")

    def __repr__(self):
        d = f", dtype={np.dtype(self.dtype).name}" if self.dtype else ""
        l = f", length={self.length!r}" if self.length else ""
        return f"MetaSpec({self.type.__name__}{d}{l})"


def write_section(g, name, data, **kw):
    """Replace /<name> with data (del first if present). Low-level write."""
    if name in g:
        del g[name]
    g.create_dataset(name, data=data, **kw)


class H5Base:
    """A thin handle over one open h5py.File: lifecycle + the generic
    attr/set_attr + meta/set_meta accessors (registry-checked, type-checked)
    + one low-level dataset write. open(path, mode="r") opens an EXISTING
    file ("r" or "r+"); create(path, mode="w") makes the parent dir, an empty
    file and the /meta group — the subclass pins its axis attrs right after
    (before any reader sees the file)."""

    # registries: ATTR_KEYS name -> type (isinstance-checked); META_KEYS
    # name -> MetaSpec (type + optional dtype / axis-0 length). Each subclass
    # declares its own, or derives from another:
    #     META_KEYS = {k: v for k, v in H5Cam.META_KEYS.items() if k != "seq"}
    ATTR_KEYS = {}
    META_KEYS = {}

    def __init__(self, path, create=False, mode=None):
        self.path = str(path)
        self._created = bool(create)
        if create:
            os.makedirs(osp.dirname(osp.abspath(self.path)), exist_ok=True)
            self._h = h5py.File(self.path, mode or "w")
            self._h.create_group("meta")
        else:
            if mode is None:
                mode = "r"
            if not osp.exists(self.path):
                raise FileNotFoundError(f"no such file: {self.path}")
            self._h = h5py.File(self.path, mode)

    def _abort_create(self):
        """Close the handle and remove a half-made file (create-mode only)."""
        self.close()
        if self._created:
            try:
                os.remove(self.path)
            except OSError:
                pass

    # ---- the handle ----
    @property
    def h(self):
        """The open h5py.File (escape hatch for raw group/attr access)."""
        return self._h

    # ---- generic attr access (checked against ATTR_KEYS) ----
    def attr(self, name):
        """attrs[name] (a known ATTR_KEYS key), or None when absent."""
        if name not in self.ATTR_KEYS:
            raise ValueError(f"unknown attr {name!r}; known: {tuple(self.ATTR_KEYS)}")
        return self._h.attrs.get(name)

    def set_attr(self, name, value):
        """Set attrs[name] (a known ATTR_KEYS key); value is checked against
        the declared type (numpy scalars accepted for the scalar types)."""
        if name not in self.ATTR_KEYS:
            raise ValueError(f"unknown attr {name!r}; known: {tuple(self.ATTR_KEYS)}")
        self._check_attr_value(name, value)
        self._h.attrs[name] = value

    # ---- generic /meta access (checked against META_KEYS) ----
    def meta(self, name):
        """/meta/<name>: str (or dict for tags, np.ndarray for frame_indices),
        None when absent."""
        if name not in self.META_KEYS:
            raise ValueError(f"unknown meta key {name!r}; known: {tuple(self.META_KEYS)}")
        m = self._h.get("meta")
        if m is None or name not in m:
            return None
        v = m[name]
        if isinstance(v, h5py.Group):
            return {k: self._as_str(v[k][()]) for k in v}
        if v.dtype.kind in ("U", "S", "O") or v.dtype == object:
            out = v[()]
            return out.decode() if isinstance(out, bytes) else out
        return v[:]

    def set_meta(self, name, value):
        """/meta/<name>: value = set/replace (str, an int64 array for
        frame_indices, or a dict[str, str] for tags), None = remove if present.
        The value is validated BEFORE the old value is deleted."""
        if name not in self.META_KEYS:
            raise ValueError(f"unknown meta key {name!r}; known: {tuple(self.META_KEYS)}")
        if value is not None:                        # validate first
            self._check_meta_spec(name, value)
        m = self._h.require_group("meta")
        if name in m:
            del m[name]
        if value is None:                            # None = remove if present
            return
        if isinstance(value, dict):                 # a /meta/<name>/ group of str
            tg = m.create_group(name)
            for k, v in value.items():
                tg.create_dataset(k, data=np.array(v, dtype=_STR))
        elif isinstance(value, np.ndarray):         # an array key (dtype-checked)
            m.create_dataset(name, data=value.astype(np.int64))
        else:                                        # a str key
            m.create_dataset(name, data=np.array(str(value), dtype=_STR))

    # ---- type checking (shared) ----
    def _check_attr_value(self, name, value):
        t = self.ATTR_KEYS[name]
        if isinstance(t, type):
            ok = _NUM_EQUIV.get(t, (t,))
            if not isinstance(value, ok):
                raise TypeError(f"attr {name!r} expects {t.__name__}, "
                                f"got {type(value).__name__}")

    def _check_meta_spec(self, name, value):
        spec = self.META_KEYS[name]
        if isinstance(spec, type):                 # a plain type (legacy)
            ok = _NUM_EQUIV.get(spec, (spec,))
            if not isinstance(value, ok):
                raise TypeError(f"meta {name!r} expects {spec.__name__}, "
                                f"got {type(value).__name__}")
            return
        spec.check(name, value, T=getattr(self, "T", None))
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(v, str):
                    raise TypeError(f"meta {name!r} value {k!r} must be a str, "
                                    f"got {type(v).__name__}")

    @staticmethod
    def _as_str(x):
        return x.decode() if isinstance(x, bytes) else x

    # ---- lifecycle ----
    def close(self):
        if self._h is not None:
            self._h.close()
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- the one low-level dataset write ----
    def write_section(self, name, data, **kw):
        """Replace /<name> with data (del first if present)."""
        write_section(self._h, name, data, **kw)
