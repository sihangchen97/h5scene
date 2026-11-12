"""The declarative section table: one Spec per fixed-shape section. This is the
single source of truth for dtype / compression / shuffle / fillvalue /
frame-2D-ness / resizability, and it drives:
  - H5Cam's set_<name> writers (dtype + filter policy)
  - H5CamRig's VDS build (dtype + fillvalue + frame2d)
  - has() (presence)
  - test_compression_policy (the pinned filter choices)
The rgb section is NOT in this table — it is variable-length image BYTES
(jpg q<=95), stored one dataset per frame, with its own policy in codecs/rgb.

The per-section ENCODING (meters->mm, mask broadcast, K/T broadcast, conf
scaling) is separate: it lives in codecs/depth (depth/depth_conf/mask) and in
codecs/pose (K/T are a plain (3,3)/(4,4) broadcast). The table is the LAYOUT;
the codecs are the CONVERSION.
"""
from collections import namedtuple

import numpy as np

# (name, dtype, compression, compression_opts, shuffle, fillvalue,
#  frame2d, resizable)
#   frame2d  : True  -> per-frame (H, W);   False -> fixed (3,3)/(4,4) frame
#   resizable: True  -> may be stored at (h,w) < (H,W), resized back on read
Spec = namedtuple("Spec",
                  "dtype compression compression_opts shuffle fillvalue frame2d resizable")

# the fixed-shape sections (rgb is handled separately as image bytes)
SECTIONS = {
    "depth":       Spec(np.uint16, "gzip", 4, True,  0,     True,  True),
    "mask":        Spec(np.bool_,  "gzip", 4, False, False, True,  False),
    "K":           Spec(np.float32, None, None, False, 0.0, False, False),
    "T":           Spec(np.float32, None, None, False, 0.0, False, False),
    "depth_conf":  Spec(np.float16, "gzip", 4, False, 0.0,  True,  True),
}

# the (3,3)/(4,4) fixed frame shape for the non-frame2d sections
FRAME_SHAPE = {"K": (3, 3), "T": (4, 4)}

# the read decoder name per section (see H5Cam._f / the rig's *_frame)
# depth & depth_conf are resized back to (H, W); mask is not
def filter_args(name):
    """The h5py create_dataset filter kwargs for a section ({} = raw)."""
    s = SECTIONS[name]
    if s.compression is None:
        return {}
    return {"compression": s.compression, "compression_opts": s.compression_opts,
            "shuffle": s.shuffle}


def has(h, name):
    """Presence of a section / meta key in an open h5py.File (robust: a raw or
    legacy file with no /rgb or /meta group does not raise)."""
    m = h.get("meta")
    has_meta = m is not None
    if name == "rgb":
        return len(h.get("rgb") or {}) > 0
    if name in SECTIONS:
        return name in h
    if has_meta:
        return name in m
    return False


def is_metric(h):
    """bool; attrs (data-side) preferred, legacy /meta/is_metric as fallback."""
    if "is_metric" in h.attrs:
        return bool(h.attrs["is_metric"])
    m = h.get("meta")
    if m is not None and "is_metric" in m:
        return bool(m["is_metric"][()])
    return False
