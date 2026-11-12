"""rgb section: jpg encoding, cv2/libjpeg quality detection (DQT fingerprint),
and rgb input normalization (array / image bytes / file paths)."""
import os
from pathlib import Path

import cv2
import numpy as np

_JPG_QUALITY = 95  # canonical rgb encoding: jpg q95 (see tests/smoke_image_formats.py)


def _jpeg_dqt_tables(data):
    """JPEG quantization tables (DQT segments) as a tuple of bytes, or None
    if data is not a JPEG file with standard 8-bit DQTs."""
    b = np.frombuffer(data, np.uint8)
    if len(b) < 4 or b[0] != 0xFF or b[1] != 0xD8:
        return None
    tables, i = [], 2
    while i + 4 <= len(b):
        m = int(b[i]) * 256 + int(b[i + 1])
        if m == 0xFFDA:  # SOS: entropy-coded scan follows, stop
            break
        ln = int(b[i + 2]) * 256 + int(b[i + 3])
        if m == 0xFFDB and ln >= 4:
            tables.append(bytes(b[i + 4:i + 4 + ln - 2].tolist()))
        i += 2 + ln
    return tuple(tables) or None


_QFP = None  # lazy fingerprint: cv2 DQT tables for a fixed set of qualities


def _qfp():
    global _QFP
    if _QFP is None:
        # cv2/libjpeg quant tables are encoder-specific (not the ISO base),
        # so match against what cv2 itself produces — exact for cv2 files.
        ref = np.random.default_rng(0).integers(0, 256, (16, 16, 3)).astype(np.uint8)
        _QFP = {q: _jpeg_dqt_tables(cv2.imencode(".jpg", ref,
                                                 [int(cv2.IMWRITE_JPEG_QUALITY), q])[1])
                for q in (1, 20, 50, 70, 90, 95, 99, 100)}
    return _QFP


def _jpeg_quality(data):
    """Detected cv2/libjpeg quality of a JPEG (int), else None (not JPEG or
    an encoder whose tables are not in the fingerprint — treat as unknown)."""
    t = _jpeg_dqt_tables(data)
    if not t:
        return None
    for q, sig in _qfp().items():
        if t == sig:
            return q
    return None


def _encode_jpg(img, quality=_JPG_QUALITY):
    return cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tobytes()


def _load_rgb_frames(flat, T, N):
    """Decode every frame, verify all frames share the first frame's (H, W),
    and apply the jpg q95 policy to the stored bytes:
      - detected cv2/libjpeg quality > 95 (or any other encoding, e.g. png)
        -> re-encode to jpg q95;
      - detected quality <= 95 -> stored verbatim (a q95 re-encode of an
        already-compressed image only adds generation, it adds no quality);
      - unknown encoder -> re-encode to jpg q95 (safe default).
    Returns (H, W, frames)."""
    H, W, out = None, None, []
    for i, b in enumerate(flat):
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"rgb frame {i} (t={i // N}, n={i % N}) is not a decodable image")
        if H is not None and img.shape[:2] != (H, W):
            raise ValueError(f"rgb frame {i} size {img.shape[:2]} != frame 0 size {(H, W)}")
        H, W = img.shape[:2]
        q = _jpeg_quality(b)
        if q is not None and q <= _JPG_QUALITY:
            out.append(bytes(b))
        else:
            out.append(_encode_jpg(img))
    return H, W, tuple(out)


def _normalize_rgb(rgb):
    """rgb -> (T, N, H, W, frames (T*N,) image bytes, all jpg q<=95).
    Input: a (T,N,H,W,3)/(H,W,3) uint8 array (encoded to jpg q95), a
    per-frame image-bytes sequence, or file paths (flat list: T frames N=1;
    (T,N) array: N cameras per frame) read from disk. See
    _load_rgb_frames for the per-frame encoding policy."""
    a = np.asarray(rgb)
    if isinstance(rgb, (str, os.PathLike)):
        a = a.reshape(1)
    # file paths: unicode array ('U'), or object array whose first element is a path
    is_paths = (a.dtype.kind == "U") or (
        a.dtype == object and a.size > 0 and isinstance(a.flat[0], (str, os.PathLike)))
    if is_paths:
        if a.ndim == 1:
            T, N = len(a), 1
            paths = [str(a[i]) for i in range(T)]
        elif a.ndim == 2:
            T, N = a.shape
            paths = [str(a[t, n]) for t in range(T) for n in range(N)]
        else:
            raise ValueError(f"rgb paths must be flat (T,) or (T,N), got ndim={a.ndim}")
        flat = tuple(Path(p).read_bytes() for p in paths)
        H, W, flat = _load_rgb_frames(flat, T, N)
        return T, N, H, W, flat
    if a.dtype.kind == "S":  # list of image bytes -> fixed-size |S dtype
        a = a.astype(object)
    if a.dtype == np.uint8:
        if a.ndim == 3:
            a = a.reshape((1, 1, *a.shape))
        if a.ndim != 5 or a.shape[4] != 3:
            raise ValueError(f"rgb array must be (T,N,H,W,3) or (H,W,3), got {a.shape}")
        T, N, H, W = a.shape[:4]
        frames = tuple(_encode_jpg(a[t, n]) for t in range(T) for n in range(N))
        return T, N, H, W, frames
    # bytes sequence
    if a.dtype != object:
        raise ValueError(f"rgb must be a uint8 array or an image-bytes sequence, got dtype {a.dtype}")
    if a.ndim == 1:
        flat = tuple(bytes(b) for b in a)
        T, N = len(flat), 1
    elif a.ndim == 2:
        T, N = a.shape
        flat = tuple(bytes(a[t, n]) for t in range(T) for n in range(N))
    else:
        raise ValueError(f"rgb bytes sequence must be flat (T,) or (T,N), got ndim={a.ndim}")
    H, W, flat = _load_rgb_frames(flat, T, N)
    return T, N, H, W, flat
