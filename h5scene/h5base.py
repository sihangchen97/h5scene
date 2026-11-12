"""H5Base: the base per-scene HDF5 FILE HANDLE (no in-memory data): the open
h5py.File, its path, lifecycle (close / context manager), the generic
attr(name) / set_attr(name, value) + meta(name) / set_meta(name, value)
accessors, and the one low-level dataset write (write_section).

Subclasses declare their OWN known-key registries as class attributes — the
base has none:
    ATTR_KEYS = {name: type, …}   # checked by attr() / set_attr()
    META_KEYS = {name: type, …}   # checked by meta() / set_meta()
the type is a Python type checked via isinstance (numpy equivalents of the
scalar types are accepted: np.bool_ for bool, np.integer for int, …); a str
label is descriptive only and never checked. (H5Cam and H5CamRig each define
what their file actually holds — e.g. the rig has no "seq".)

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

    # registries: name -> expected type (a Python type checked via isinstance,
    # or a str label = descriptive only). Each subclass declares its own, or
    # extends the base:  ATTR_KEYS = {**H5Base.ATTR_KEYS, "X": bool}
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
        self._check_type(f"attr {name!r}", self.ATTR_KEYS[name], value)
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
            self._check_type(f"meta {name!r}", self.META_KEYS[name], value)
            if isinstance(value, dict):
                for k, v in value.items():
                    if not isinstance(v, str):
                        raise TypeError(f"meta {name!r} value {k!r} must be a str, "
                                        f"got {type(v).__name__}")
        m = self._h.require_group("meta")
        if name in m:
            del m[name]
        if value is None:                            # None = remove if present
            return
        if isinstance(value, dict):                 # a /meta/<name>/ group of str
            tg = m.create_group(name)
            for k, v in value.items():
                tg.create_dataset(k, data=np.array(v, dtype=_STR))
        elif isinstance(value, np.ndarray):         # frame_indices (int64 (T,))
            if value.ndim != 1:
                raise ValueError(f"{name} must be (T,), got {value.shape}")
            m.create_dataset(name, data=value.astype(np.int64))
        else:                                        # a str key
            m.create_dataset(name, data=np.array(str(value), dtype=_STR))

    # ---- type checking (shared) ----
    @classmethod
    def _check_type(cls, what, expected, value):
        if not isinstance(expected, type):          # a "str" label: descriptive only
            return
        ok = _NUM_EQUIV.get(expected, (expected,))
        if not isinstance(value, ok):
            raise TypeError(f"{what} expects {expected.__name__}, "
                            f"got {type(value).__name__}")

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
