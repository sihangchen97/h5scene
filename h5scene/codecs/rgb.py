"""rgb section: jpg encoding + cv2/libjpeg quality detection (DQT
fingerprint) + rgb input normalization, ONE entry per input kind
(normalize_img_array / normalize_img_bytes / normalize_img_paths). THE ONE
place cv2's colour order lives: public API in and out is RGB; cv2's BGR is
converted at the boundary (encode: RGB -> BGR; decode: BGR -> RGB), so the
stored jpgs carry RGB in every tool that reads them and a red png file
input reads back red.
"""
import os
from pathlib import Path

import cv2
import numpy as np

JPG_QUALITY = 95  # canonical rgb encoding: jpg q95 (see tests/smoke_image_formats.py)


# ---- the DQT fingerprint (cv2/libjpeg quality detection) ----
def jpeg_dqt_tables(data):
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
        _QFP = {q: jpeg_dqt_tables(cv2.imencode(".jpg", ref,
                                                [int(cv2.IMWRITE_JPEG_QUALITY), q])[1])
                for q in (1, 20, 50, 70, 90, 95, 99, 100)}
    return _QFP


def jpeg_quality(data):
    """Detected cv2/libjpeg quality of a JPEG (int), else None.

    None means: not a JPEG, or a JPEG whose DQT tables are not in the
    cv2 fingerprint (a non-cv2 encoder, or a cv2 quality we didn't
    fingerprint, e.g. q80) — stored/kept bytes are only verbatim for
    q ∈ {1, 20, 50, 70, 90, 95}; everything else is re-encoded at q95."""
    t = jpeg_dqt_tables(data)
    if not t:
        return None
    for q, sig in _qfp().items():
        if t == sig:
            return q
    return None


# ---- encode / decode (the colour-order boundary) ----
def encode_rgb(img, quality=JPG_QUALITY):
    """One RGB (H, W, 3) uint8 -> jpg bytes (cv2 encodes BGR internally)."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok or buf is None:
        raise ValueError(f"cv2.imencode failed for an image of shape {img.shape}")
    return buf.tobytes()


def decode_rgb(b):
    """jpg/image bytes -> RGB (H, W, 3) uint8 (cv2 decodes BGR internally)."""
    img = cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a decodable image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def jpeg_hw(data):
    """(H, W) of a JPEG from its SOF marker alone (no decode): None if the
    data is not a JPEG (or has no SOF before SOS). For cheap size checks."""
    b = np.frombuffer(bytes(data), np.uint8)
    if len(b) < 4 or b[0] != 0xFF or b[1] != 0xD8:
        return None
    i = 2
    while i + 4 <= len(b):
        m = int(b[i]) * 256 + int(b[i + 1])
        if m == 0xFFDA:                                 # SOS: no SOF seen
            return None
        # SOF0-SOF15 EXCEPT C4 (DHT), C8 (JPG) and CC (DAC) — those are
        # other markers with a different layout, and some encoders put a DHT
        # before the SOF; treating them as SOF would read a wrong size.
        if m in (0xFFC0, 0xFFC1, 0xFFC2, 0xFFC3, 0xFFC5, 0xFFC6, 0xFFC7,
                 0xFFC9, 0xFFCA, 0xFFCB, 0xFFCD, 0xFFCE, 0xFFCF):
            # SOF segment: len(2) precision(1) H(2) W(2) ncomp(1) — at i+4..i+9
            if i + 10 > len(b):                         # SOF present but truncated
                return None
            return int(b[i + 5]) * 256 + int(b[i + 6]), int(b[i + 7]) * 256 + int(b[i + 8])
        ln = int(b[i + 2]) * 256 + int(b[i + 3])
        if ln < 4:                                      # malformed: skip the marker
            return None
        i += 2 + ln
    return None


# ---- input normalization: ONE entry per input kind (array / image bytes /
# file paths); each takes ONLY its kind and raises a clear error otherwise ----
def _normalize_frames(flat):
    """(H, W, frames) from a non-empty tuple of image bytes, all jpg q<=95:
    a frame whose detected cv2/libjpeg quality <= 95 is kept VERBATIM (the
    SOF header gives its size); quality > 95, another encoding, or an unknown
    encoder is re-encoded to jpg q95 (the only full decode). All frames must
    share the first frame's size."""
    H, W, out = None, None, []
    for i, b in enumerate(flat):
        b = bytes(b)
        q = jpeg_quality(b)
        if q is not None and q <= JPG_QUALITY:
            hw = jpeg_hw(b)                        # SOF header only, no decode
            if hw is None:                         # a jpeg without a SOF: full decode
                hw = decode_rgb(b).shape[:2]
            if H is not None and hw != (H, W):
                raise ValueError(f"rgb frame {i} size {hw} != frame 0 size {(H, W)}")
            H, W = hw
            out.append(b)                          # stored verbatim
        else:                                      # re-encode: the only full decode
            img = decode_rgb(b)                    # RGB
            if H is not None and img.shape[:2] != (H, W):
                raise ValueError(f"rgb frame {i} size {img.shape[:2]} != frame 0 size {(H, W)}")
            H, W = img.shape[:2]
            out.append(encode_rgb(img))
    return H, W, tuple(out)


def _frames_from_bytes(frames):
    """a sequence of image bytes (T,) / (T,N,) -> a flat tuple of bytes."""
    if isinstance(frames, (list, tuple)) and len(frames) == 0:
        raise ValueError("rgb bytes is an empty frame sequence (no frames to store)")
    a = np.asarray(frames)
    if a.dtype.kind == "S":                        # list of image bytes -> |S dtype
        a = a.astype(object)
    if a.dtype != object:
        raise ValueError(f"rgb bytes must be a sequence of image bytes, got dtype {a.dtype}")
    if a.ndim == 1:
        return tuple(bytes(b) for b in a)
    if a.ndim == 2:
        return tuple(bytes(a[t, n]) for t in range(a.shape[0]) for n in range(a.shape[1]))
    raise ValueError(f"rgb bytes must be a flat (T,) or (T,N) sequence, got ndim={a.ndim}")


def normalize_img_array(rgb):
    """a uint8 RGB ARRAY (T,N,H,W,3) / (T,H,W,3) / an (H,W,3) frame (T=1, N=1)
    -> (H, W, frames (T*N,) jpg q95). ARRAYS ONLY (image bytes / file paths:
    the normalize_img_bytes / normalize_img_paths entries)."""
    a = np.asarray(rgb)
    if a.dtype != np.uint8:
        raise TypeError(f"rgb must be a uint8 RGB array, got {a.dtype.name}")
    if a.ndim == 3:
        a = a[None, None]
    elif a.ndim == 4:
        a = a[:, None]
    if a.ndim != 5 or a.shape[4] != 3:
        raise ValueError(f"rgb array must be (T,N,H,W,3), (T,H,W,3) or (H,W,3), got {a.shape}")
    T, N, H, W = a.shape[:4]
    flat = tuple(encode_rgb(a[t, n]) for t in range(T) for n in range(N))
    return H, W, flat


def normalize_img_bytes(frames):
    """a sequence of image BYTES (T,) / (T,N,) -> (H, W, frames (T*N,), all
    jpg q<=95): the per-frame policy above (q<=95 verbatim, else re-encoded
    q95). BYTES ONLY (arrays / file paths: the other entries). An empty
    sequence is a clear error."""
    return _normalize_frames(_frames_from_bytes(frames))


def normalize_img_paths(paths):
    """FILE PATHS (a str/Path, or a flat (T,) / (T,N) sequence) read from disk
    -> (H, W, frames (T*N,), all jpg q<=95, the same per-frame policy as
    normalize_img_bytes). PATHS ONLY (arrays / bytes: the other entries). An
    empty path sequence is a clear error."""
    a = np.asarray(paths)
    if isinstance(paths, (str, os.PathLike)):
        a = a.reshape(1)
    is_paths = (a.dtype.kind == "U") or (
        a.dtype == object and a.size > 0 and isinstance(a.flat[0], (str, os.PathLike)))
    if not is_paths:
        raise ValueError(f"rgb paths must be file paths, got dtype {a.dtype}")
    if a.ndim == 1:
        flat = tuple(Path(str(a[i])).read_bytes() for i in range(a.size))
    elif a.ndim == 2:
        flat = tuple(Path(str(a[t, n])).read_bytes()
                     for t in range(a.shape[0]) for n in range(a.shape[1]))
    else:
        raise ValueError(f"rgb paths must be flat (T,) or (T,N), got ndim={a.ndim}")
    if not flat:
        raise ValueError("rgb paths is an empty frame sequence (no frames to store)")
    return _normalize_frames(flat)
