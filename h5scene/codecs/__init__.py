"""codecs: the per-section encode/decode policies (one code path each).

- rgb.py     — jpg q<=95 encoding + cv2 colour-order boundary (public API is
               RGB; cv2's BGR is converted at encode/decode) + DQT quality
               detection + rgb input normalization (array / bytes / paths).
- depth.py   — mm-only / meters-only conversion (the unit is fixed by the
               entry point, never guessed from the dtype), the 0/65535 sentinel
               rules, the (h,w) aspect-ratio + resize rules, depth_conf,
               the derived depth_mask.
- pose.py    — K (3,3) / T (4,4) broadcast.
- sections.py— the declarative SECTIONS table (dtype / compression / shuffle /
               fillvalue / frame2d) + has() + is_metric; it drives
               the H5Cam writers, the rig's VDS build, and the tests.
"""
from .rgb import (JPG_QUALITY, decode_rgb, encode_rgb, jpeg_dqt_tables,
                  jpeg_hw, jpeg_quality, normalize_rgb)
from .depth import (MM_MAX, conf_frame, depth_hw_ok, depth_mask, meters_frame,
                    mm_frame, to_conf, to_mask, to_mm, to_meters_mm)
from .pose import broadcast_matrix
from .sections import FRAME_SHAPE, SECTIONS, Spec, filter_args, has, is_metric

__all__ = [
    "JPG_QUALITY", "decode_rgb", "encode_rgb", "jpeg_dqt_tables", "jpeg_quality",
    "jpeg_hw", "normalize_rgb",
    "MM_MAX", "conf_frame", "depth_hw_ok", "depth_mask", "meters_frame", "mm_frame",
    "to_conf", "to_mask", "to_mm", "to_meters_mm",
    "broadcast_matrix",
    "FRAME_SHAPE", "SECTIONS", "Spec", "filter_args", "has", "is_metric",
]
