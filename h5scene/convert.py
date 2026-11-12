"""Input-shape conversion helpers: mask / depth / depth_conf / K / pose, and
the (h, w) < (H, W) depth aspect-ratio + resize rules shared by H5Cam and H5CamRig."""
import cv2
import numpy as np


def _frame_keys(T, N):
    """Derived rgb keys: one per (t, n), row-major f"{t}_{n}". No per-frame ids stored."""
    return [f"{t}_{n}" for t in range(T) for n in range(N)]


def _broadcast(a, dtype, T, N, tail):
    """(…tail) broadcast to every (t,n) or (T,N,…tail) as-is -> (T,N,…tail)."""
    a = np.asarray(a)
    if a.ndim == len(tail):
        a = np.broadcast_to(a, (T, N) + tail)
    else:
        a = a.reshape((T, N) + tail)
    try:
        a = np.broadcast_to(a, (T, N) + tail)
    except ValueError:
        raise ValueError(f"shape {a.shape} does not broadcast to ({T}, {N}, {tail})")
    return a.astype(dtype, copy=False)


def _to_mask(mask, T, N, H, W):
    """bool/uint8 mask -> (T,N,H,W) bool (1 = valid, 0 = padding), FULL (H, W).
    Input: (T,N,H,W) pixel-wise, (H,W) broadcast to every (t,n), or (T,N)
    per-frame (which slots are padding).

    The (T,N) per-slot and (H,W) broadcast shapes are checked (T,N)-first.
    When T == H and N == W the two are indistinguishable, so a 2-D input is
    rejected — pass an explicit (T,N,H,W) pixel mask instead."""
    a = np.asarray(mask)
    if a.ndim == 2 and a.shape == (T, N):
        if a.shape == (H, W):
            raise ValueError(
                f"(T,N) == (H,W) == {tuple(a.shape)}: ambiguous per-slot vs pixel mask; "
                "pass an explicit (T,N,H,W) pixel mask")
        a = a.astype(bool)[:, :, None, None]
        try:
            a = np.broadcast_to(a, (T, N, H, W))
        except ValueError:
            raise ValueError(f"(T,N) mask does not broadcast to ({T}, {N}, {H}, {W})")
        return a.copy()
    if a.ndim == 2 and a.shape == (H, W):
        a = a.reshape((1, 1, H, W))
    try:
        a = np.broadcast_to(a, (T, N, H, W))
    except ValueError:
        raise ValueError(f"mask shape {a.shape} does not broadcast to ({T}, {N}, {H}, {W})")
    return (a != 0).astype(bool).copy()


def _to_depth_conf(conf, T, N):
    """0-1 floats (or 0-65535 uint16) -> (T,N,h,w) fp16 0-1.
    (h,w) may be smaller than the scene (H,W) as long as the aspect ratio
    H/W matches (see _depth_hw_ok); depth and depth_conf must be stored
    at the same size. (H,W) input is reshaped to a single (1,1,h,w)."""
    a = np.asarray(conf)
    if a.ndim == 2:
        a = a.reshape((1, 1, *a.shape))
    if a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    a = np.clip(a, 0.0, 1.0)
    a = np.where(np.isfinite(a), a, 0.0)
    try:
        a = np.broadcast_to(a, (T, N, a.shape[2], a.shape[3]))
    except ValueError:
        raise ValueError(f"depth_conf shape {a.shape} does not broadcast to (T={T}, N={N})")
    return a.astype(np.float16).copy()


def _to_depth_mm(depth, T, N):
    """float meters (or uint16 mm) -> (T,N,h,w) uint16 mm; d > 65.535 m -> 0 (invalid).
    (h,w) may be smaller than the scene (H,W) as long as the aspect ratio H/W
    matches (see _depth_hw_ok); (H,W) input is reshaped to a single (1,1,h,w)."""
    a = np.asarray(depth)
    if a.ndim == 2:
        a = a.reshape((1, 1, *a.shape))
    mm = a if a.dtype == np.uint16 else (a.astype(np.float32) * 1000.0)
    mm = np.rint(np.where(np.isfinite(mm), mm, 0.0))
    mm = np.where(mm > 65535, 0, mm)  # too far -> invalid 0, never saturated 65535
    try:
        mm = np.broadcast_to(mm, (T, N, mm.shape[2], mm.shape[3]))
    except ValueError:
        raise ValueError(f"depth shape {mm.shape} does not broadcast to (T={T}, N={N})")
    return np.clip(mm, 0, 65535).astype(np.uint16).copy()


def _depth_hw_ok(h, w, H, W):
    """(h, w) usable for a scene of (H, W): same aspect ratio and not larger."""
    return h <= H and w <= W and h * W == w * H


def _resize_hw(frame, H, W, interp):
    """(h, w) frame -> (H, W); resizes only when the stored size differs."""
    if frame.shape[:2] == (H, W):
        return frame
    return cv2.resize(frame, (W, H), interpolation=interp)
