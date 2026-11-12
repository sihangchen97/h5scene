"""H5Base: the base per-scene HDF5 FILE HANDLE (no in-memory data): the open
h5py.File, its path, lifecycle (close / context manager), the generic
attr(name) / set_attr(name, value) + meta(name) / set_meta(name, value)
accessors, and the one low-level dataset write (write_section).

Subclasses declare their OWN known-key registries as class attributes — the
base has none:
    ATTR_KEYS = (("name", "type"), …)   # checked by attr() / set_attr()
    META_KEYS = (("name", "type"), …)   # checked by meta() / set_meta()
(H5Cam and H5CamRig each define what their file actually holds — e.g. the
rig has no "seq"; H5Cam does.) Nothing scene-specific — no shape: the axis
convention (T,H,W vs T,N,Hmax,Wmax) and the section semantics live on the
subclasses, which build their set_*/read methods on the shared helpers in
_io.py:

    H5Cam(H5Base)     — single-camera (T, H, W), attrs.THW
    H5CamRig(H5Base)  — multi-camera (T, N, Hmax, Wmax), references only
"""
import os
import os.path as osp

import h5py
import numpy as np

from ._io import _STR, write_section


class H5Base:
    """A thin handle over one open h5py.File: lifecycle + the generic
    attr/set_attr + meta/set_meta accessors (checked against the subclass's
    ATTR_KEYS / META_KEYS) + one low-level dataset write. Open mode opens an
    EXISTING file with 'r' (a missing path raises FileNotFoundError rather
    than creating an orphan .h5); create mode ('w') makes the parent dir, an
    empty file and the /meta group — subclasses pin their axis attrs right
    after (before any reader sees the file)."""

    # registries: name -> expected type (a Python type checked via isinstance,
    # or a str label = descriptive only). Each subclass declares its own, or
    # extends the base:  self.ATTR_KEYS = {**type(self).ATTR_KEYS, "X": bool}
    ATTR_KEYS = {}
    META_KEYS = {}

    def __init__(self, path, create=False):
        self.path = str(path)
        if create:
            os.makedirs(osp.dirname(osp.abspath(self.path)), exist_ok=True)
            self._h = h5py.File(self.path, "w")
            self._h.create_group("meta")
        else:
            self._h = h5py.File(self.path, "r")

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
        the declared type when it is a Python type."""
        if name not in self.ATTR_KEYS:
            raise ValueError(f"unknown attr {name!r}; known: {tuple(self.ATTR_KEYS)}")
        t = self.ATTR_KEYS[name]
        if isinstance(t, type) and not isinstance(value, t):
            raise TypeError(f"attr {name!r} expects {t.__name__}, got {type(value).__name__}")
        self._h.attrs[name] = value

    # ---- generic /meta access (checked against META_KEYS) ----
    def meta(self, name):
        """/meta/<name>: str (or dict for tags, np.ndarray for frame_indices),
        None when absent."""
        if name not in self.META_KEYS:
            raise ValueError(f"unknown meta key {name!r}; known: {tuple(self.META_KEYS)}")
        m = self._h["meta"]
        if name not in m:
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
        frame_indices, or a dict for tags), None = remove if present."""
        if name not in self.META_KEYS:
            raise ValueError(f"unknown meta key {name!r}; known: {tuple(self.META_KEYS)}")
        m = self._h.require_group("meta")
        if name in m:
            del m[name]
        if value is None:                            # None = remove if present
            return
        t = self.META_KEYS[name]
        if isinstance(t, type) and not isinstance(value, t):
            raise TypeError(f"meta {name!r} expects {t.__name__}, got {type(value).__name__}")
        if isinstance(value, dict):                 # a /meta/<name>/ group of str
            tg = m.create_group(name)
            for k, v in value.items():
                tg.create_dataset(k, data=np.array(str(v), dtype=_STR))
        elif isinstance(value, np.ndarray):         # frame_indices (int64 (T,))
            if value.ndim != 1:
                raise ValueError(f"{name} must be (T,), got {value.shape}")
            m.create_dataset(name, data=value.astype(np.int64))
        else:                                        # a str key
            m.create_dataset(name, data=np.array(str(value), dtype=_STR))

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
