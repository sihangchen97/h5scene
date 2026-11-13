"""The declarative section table: one Spec per fixed-shape section. This is the
single source of truth for dtype / compression / shuffle / fillvalue /
frame-2D-ness, and it drives:
  - H5Cam's set_<name> writers (dtype + filter policy)
  - H5CamRig's VDS build (dtype + fillvalue + frame2d)
  - has() (presence)
  - test_compression_policy (the pinned filter choices)
The img section is NOT in this table — it is variable-length image BYTES
(jpg q<=95), stored one dataset per frame, with its own policy in codecs/rgb.

The per-section CONVERSION (meters->mm, mask broadcast, K/T broadcast, the
0-1 conf guard) is separate: it lives in codecs/depth (depth/depth_conf/mask) and in
codecs/pose (K/T are a plain (3,3)/(4,4) broadcast). The table is the LAYOUT;
the codecs are the CONVERSION.
"""
from collections import namedtuple

import numpy as np

# (name, dtype, compression, compression_opts, shuffle, fillvalue,
#  frame2d)
#   frame2d: True -> per-frame (H, W);   False -> fixed (3,3)/(4,4) frame
Spec = namedtuple("Spec",
                  "dtype compression compression_opts shuffle fillvalue frame2d")

# the fixed-shape sections (rgb is handled separately as image bytes)
SECTIONS = {
    "depth":       Spec(np.uint16, "gzip", 4, True,  0,     True),
    "mask":        Spec(np.bool_,  "gzip", 4, False, False, True),
    "K":           Spec(np.float32, None, None, False, 0.0, False),
    "T":           Spec(np.float32, None, None, False, 0.0, False),
    "depth_conf":  Spec(np.float16, "gzip", 4, False, 0.0,  True),
}

# the (3,3)/(4,4) fixed frame shape for the non-frame2d sections
FRAME_SHAPE = {"K": (3, 3), "T": (4, 4)}

# ---- H5Geo (static world-frame geometry; requirements_gt_geometry) ----------
# (level, dtype, cols, compression, compression_opts, shuffle, chunk_rows)
#   level: the owning Houdini group — points (M rows) / vertices (V, corners)
#          / prims (F); a standard name fixes its dtype, cols and meaning.
#   cols: 1 = stored 1-D, (N,) — otherwise (N, cols).
#   chunk_rows: rows per chunk (the sweep pinned 262144 for the (M,3) point
#          attributes, 65536 for ptnum; unswept rows default to 65536).
# Filter + chunk policy (the sweep in tests/probes/geo/README.md, run on the
# 5 sample plys, projected 43-scene total 811 MB < 1.8 GB): every multi-byte
# type (P, N, ptnum, …) gzip4 + SHUFFLE — the C1 pick, deliberately kept
# consistent with /depth and the library convention although the sweep
# measured gzip6+shuffle marginally faster for P/N (owner decision, recorded
# in tests/probes/geo/README.md); 1-byte types (Cd, …) plain gzip4 (shuffle
# has nothing to split, same reasoning as 'mask').
GeoSpec = namedtuple("GeoSpec",
                     "level dtype cols compression compression_opts shuffle chunk_rows")

GEO_ATTRS = {
    "points/P":          GeoSpec("points",   np.float32, 3, "gzip", 4, True, 262_144),
    "points/Cd":         GeoSpec("points",   np.uint8,   3, "gzip", 4, False, 262_144),
    "points/N":          GeoSpec("points",   np.float32, 3, "gzip", 4, True, 262_144),
    "points/Alpha":      GeoSpec("points",   np.uint8,   1, "gzip", 4, False, 65_536),
    "points/uv":         GeoSpec("points",   np.float32, 2, "gzip", 4, True, 65_536),
    "points/pscale":     GeoSpec("points",   np.float32, 1, "gzip", 4, True, 65_536),
    "points/conf":       GeoSpec("points",   np.float32, 1, "gzip", 4, True, 65_536),
    "points/intensity":  GeoSpec("points",   np.float32, 1, "gzip", 4, True, 65_536),
    "points/class":      GeoSpec("points",   np.int32,   1, "gzip", 4, True, 65_536),
    "points/id":         GeoSpec("points",   np.int32,   1, "gzip", 4, True, 65_536),
    "vertices/ptnum":    GeoSpec("vertices", np.int32,   1, "gzip", 4, True, 65_536),
    "vertices/N":        GeoSpec("vertices", np.float32, 3, "gzip", 4, True, 65_536),
    "vertices/uv":       GeoSpec("vertices", np.float32, 2, "gzip", 4, True, 65_536),
    "vertices/Cd":       GeoSpec("vertices", np.uint8,   3, "gzip", 4, False, 65_536),
    "prims/vertexcount": GeoSpec("prims",    np.int32,   1, "gzip", 4, True, 65_536),
    "prims/N":           GeoSpec("prims",    np.float32, 3, "gzip", 4, True, 65_536),
    "prims/Cd":          GeoSpec("prims",    np.uint8,   3, "gzip", 4, False, 65_536),
    "prims/material":    GeoSpec("prims",    np.int32,   1, "gzip", 4, True, 65_536),
    "prims/class":       GeoSpec("prims",    np.int32,   1, "gzip", 4, True, 65_536),
    "prims/id":          GeoSpec("prims",    np.int32,   1, "gzip", 4, True, 65_536),
}

# allowed dtypes for a CUSTOM attribute (D10): multi-byte -> gzip4+shuffle,
# 1-byte (u8/bool) -> plain gzip4
CUSTOM_DTYPES = (np.float32, np.float16, np.int32, np.uint8, bool)

def geo_chunks(rows, cols, chunk_rows=65_536):
    """The H5Geo chunk shape for rows rows of cols columns (1-D when cols 1),
    capped at the spec row's chunk_rows."""
    c = min(rows, chunk_rows)
    return (c,) if cols == 1 else (c, cols)


def filter_args(name, table=SECTIONS):
    """The h5py create_dataset filter kwargs for one table row ({} = raw).
    table: SECTIONS (the camera fixed-shape sections) or GEO_ATTRS (the
    H5Geo per-level attributes, keyed 'points/P' etc.)."""
    s = table[name]
    if s.compression is None:
        return {}
    return {"compression": s.compression, "compression_opts": s.compression_opts,
            "shuffle": s.shuffle}


def has(h, name, table=SECTIONS):
    """Whether one table row (or the img section) is present in h;
    other names (meta keys, groups) are False — use `meta(name) is not None`
    for meta keys. /img may be absent (an image-less camera or rig).
    table: SECTIONS (the camera sections) or GEO_ATTRS (keyed 'points/P' —
    h5py path lookups work on the nested groups)."""
    if name == "img":
        return len(h.get("img") or []) > 0
    return name in table and name in h


def is_metric(h):
    """bool, read from attrs (data-side) — always written by create/build,
    so a KeyError here means a file that is not ours. No legacy fallback."""
    return bool(h.attrs["is_metric"])
