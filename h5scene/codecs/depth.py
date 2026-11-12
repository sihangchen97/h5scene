"""depth section: meters<->mm conversion, the sentinel rules, the (h,w)
aspect-ratio + resize rules, depth_conf, and the derived depth_mask.

Sentinels: BOTH 0 AND 65535 mean invalid (mm). The writer clips
d > 65.535 m -> 0 (never saturates to 65535); a read re-clips >= 65535 -> 0
(double safety). Integer dtypes are already mm; floats are meters.
"""
import cv2
import numpy as np

MM_MAX = 65535


# ---- input -> stored (T, h, w) ----
def to_mm(depth, T):
    """(T, h, w) float METERS (or integer MM) -> (T, h, w) uint16 mm.
    d > 65.535 m -> 0 (invalid); never saturated to 65535."""
    a = np.asarray(depth)
    if a.dtype.kind in "iu":                       # integers are already mm
        mm = a.astype(np.float64)
    else:                                          # floats are meters
        mm = a.astype(np.float64) * 1000.0
    mm = np.rint(np.where(np.isfinite(mm), mm, 0.0))
    mm = np.where(mm > MM_MAX, 0, mm)
    return np.clip(mm, 0, MM_MAX).astype(np.uint16)


def to_conf(conf, T):
    """(T, h, w) 0-1 floats (or 0-65535 uint16) -> (T, h, w) f16 0-1."""
    a = np.asarray(conf)
    if a.dtype == np.uint16:
        a = a.astype(np.float64) / 65535.0
    a = np.clip(a, 0.0, 1.0)
    a = np.where(np.isfinite(a), a, 0.0)
    return a.astype(np.float16)


def to_mask(mask, T, H, W):
    """bool/uint8 -> (T, H, W) bool (1 = valid, 0 = padding). Input: (T,H,W)
    per pixel, (H,W) broadcast to every frame, or (T,) per-frame slots."""
    a = np.asarray(mask)
    if a.ndim == 3:
        a = np.broadcast_to(a, (T, H, W))
    elif a.ndim == 2 and a.shape == (H, W):
        a = np.broadcast_to(a, (T, H, W))
    elif a.ndim == 1 and a.shape == (T,):
        a = np.broadcast_to(a[:, None, None], (T, H, W))
    else:
        raise ValueError(f"mask shape {a.shape} does not broadcast to "
                         f"({T}, {H}, {W}) / ({H}, {W}) / ({T},)")
    return (a != 0).astype(bool)


# ---- stored (h, w) frame -> scene (H, W) ----
def depth_hw_ok(h, w, H, W):
    """(h, w) usable for a scene of (H, W): same aspect ratio and not larger."""
    return h <= H and w <= W and h * W == w * H


def _resize(frame, H, W):
    if frame.shape[:2] == (H, W):
        return frame
    return cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)


def mm_frame(frame, H, W):
    """One stored (h, w) uint16 mm frame -> (H, W) uint16 (nearest; the
    >= 65535 -> 0 double-safety is re-applied)."""
    f = _resize(frame, H, W).astype(np.uint16)
    return np.where(f >= MM_MAX, 0, f)


def meters_frame(frame, H, W):
    """One stored (h, w) uint16 mm frame -> (H, W) float32 meters."""
    return (mm_frame(frame, H, W).astype(np.int32) / 1000.0).astype(np.float32)


def conf_frame(frame, H, W):
    """One stored (h, w) f16 0-1 frame -> (H, W) float32 0-1."""
    return _resize(frame, H, W).astype(np.float32)


def depth_mask(mm, conf, conf_min):
    """(H, W) bool usable-depth mask (derived on read): mm not an invalid
    sentinel (0 AND 65535 excluded) and, if conf given, conf > (0 if
    conf_min is None, else conf_min)."""
    valid = (mm != 0) & (mm != MM_MAX)
    if conf is not None:
        thr = 0.0 if conf_min is None else conf_min
        valid &= conf > thr
    return valid
