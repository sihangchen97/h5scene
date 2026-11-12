"""depth section: mm-only / meters-only conversion, the sentinel rules, the
(h,w) aspect-ratio + resize rules, depth_conf, and the derived depth_mask.

Sentinels: BOTH 0 AND 65535 mean invalid (mm). The meters writer clips
d >= 65.535 m -> 0 (never saturates to 65535); a read re-clips >= 65535 -> 0
(double safety). The unit is fixed by the ENTRY POINT (set_depth_mm takes
integers, set_depth_meters takes floats) — never guessed from the dtype.
"""
import cv2
import numpy as np

MM_MAX = 65535


# ---- input -> stored (T, h, w) ----
def to_mm(mm,):
    """(T, h, w) INTEGER mm -> (T, h, w) uint16 mm (unit guard: an integer
    dtype only — a float means meters and must go through to_meters_mm)."""
    a = np.asarray(mm)
    if a.dtype.kind not in "iu":
        raise TypeError(f"set_depth_mm wants integer mm, got {a.dtype.name} — "
                        f"for METERS use set_depth_meters")
    v = a.astype(np.float64)
    v = np.rint(np.where(np.isfinite(v), v, 0.0))
    v = np.where(v >= MM_MAX, 0, v)                # round-trip: 65535 = invalid
    return np.clip(v, 0, MM_MAX).astype(np.uint16)


def to_meters_mm(meters,):
    """(T, h, w) float METERS -> (T, h, w) uint16 mm (unit guard: a float
    dtype only — an integer means mm and must go through to_mm).
    d >= 65.535 m -> 0 (invalid): 65535 is the invalid sentinel, so the max
    representable depth can never survive a round trip as "valid" — write and
    read agree on the sentinel (65535 is never a valid depth)."""
    a = np.asarray(meters)
    if a.dtype.kind != "f":
        raise TypeError(f"set_depth_meters wants float METERS, got {a.dtype.name} "
                        f"— for integer MM use set_depth_mm")
    v = a.astype(np.float64) * 1000.0
    v = np.rint(np.where(np.isfinite(v), v, 0.0))
    v = np.where(v >= MM_MAX, 0, v)                # round-trip: 65535 = invalid
    return np.clip(v, 0, MM_MAX).astype(np.uint16)


def to_conf(conf,):
    """(T, h, w) float 0-1 -> (T, h, w) f16 0-1 (range guard: a float dtype
    only — an integer is never a 0-1 confidence and is rejected)."""
    a = np.asarray(conf)
    if a.dtype.kind != "f":
        raise TypeError(f"set_depth_conf_01 wants float 0-1, got {a.dtype.name} "
                        f"— integers are not accepted")
    if a.size and (a.min() < 0.0 or a.max() > 1.0):
        raise ValueError(f"depth_conf must be 0-1, got "
                         f"[{a.min():.6g}, {a.max():.6g}]")
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
