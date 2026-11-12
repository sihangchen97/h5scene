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

def filter_args(name):
    """The h5py create_dataset filter kwargs for a section ({} = raw)."""
    s = SECTIONS[name]
    if s.compression is None:
        return {}
    return {"compression": s.compression, "compression_opts": s.compression_opts,
            "shuffle": s.shuffle}


def has(h, name):
    """Whether one of the SECTIONS (or the img section) is present in h;
    other names (meta keys, groups) are False — use `meta(name) is not None`
    for meta keys. /img may be absent (an image-less camera or rig)."""
    if name == "img":
        return len(h.get("img") or []) > 0
    return name in SECTIONS and name in h


def is_metric(h):
    """bool, read from attrs (data-side) — always written by create/build,
    so a KeyError here means a file that is not ours. No legacy fallback."""
    return bool(h.attrs["is_metric"])
