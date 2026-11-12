"""Tests for codecs/rgb.py — the jpg q<=95 encoding policy, the RGB colour
boundary, and the input normalisation (ISSUE #35: one test file per codecs
module). Runs in the default test env (numpy + opencv only, no pillow)."""
import cv2
import numpy as np
import pytest

from h5scene.codecs import (JPG_QUALITY, decode_rgb, encode_rgb, jpeg_hw,
                           jpeg_quality, normalize_rgb)


def _red(h=8, w=12):
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = 200                       # a RED image (R channel high)
    return img


def _png_bytes(tmp_path, img):
    p = str(tmp_path / "f.png")
    cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return bytes(open(p, "rb").read())


def test_encode_decode_roundtrip_rgb_order(tmp_path):
    # the stored jpg keeps the R channel high for a red image (ISSUE #8: the
    # public API is RGB end to end; cv2's BGR is converted at the boundary)
    b = encode_rgb(_red())
    back = decode_rgb(b)
    assert back[0, 0, 0] > back[0, 0, 2]            # still red on read-back
    assert jpeg_quality(b) == JPG_QUALITY
    # a red png FILE decodes back red (the public API reads RGB, ISSUE #8)
    back = decode_rgb(_png_bytes(tmp_path, _red()))
    assert back[0, 0, 0] > back[0, 0, 2]


def test_normalize_keeps_low_quality_jpeg_verbatim():
    ok, buf = cv2.imencode(".jpg", _red(), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    b = bytes(buf.tobytes())
    _, _, (out,) = normalize_rgb([b])        # q50 <= 95 -> stored as-is
    assert out == b


def test_normalize_reencodes_over_95_jpeg():
    ok, buf = cv2.imencode(".jpg", _red(), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    b = bytes(buf.tobytes())
    _, _, (out,) = normalize_rgb([b])        # q100 > 95 -> re-encoded q95
    assert out != b and jpeg_quality(out) == JPG_QUALITY


def test_normalize_reencodes_png(tmp_path):
    _, _, (out,) = normalize_rgb([_png_bytes(tmp_path, _red())])
    assert jpeg_quality(out) == JPG_QUALITY        # a png became a q95 jpg
    assert decode_rgb(out)[0, 0, 0] > decode_rgb(out)[0, 0, 2]


def test_jpeg_quality_cv2_fingerprints():
    # the DQT fingerprint only knows cv2's quality ladder; an unknown encoder
    # (e.g. a real q80) comes back None -> re-encoded (ISSUE #14's docstring)
    for q in (1, 20, 50, 70, 90, 95, 99, 100):
        ok, buf = cv2.imencode(".jpg", _red(), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        assert jpeg_quality(bytes(buf.tobytes())) == q


def test_jpeg_hw_from_sof_header_only():
    b = encode_rgb(_red(8, 12))
    assert jpeg_hw(b) == (8, 12)
    assert jpeg_hw(b[:10]) is None                  # truncated: no SOF yet


def test_jpeg_hw_survives_truncated_and_odd_headers():
    # #44: a truncated SOF (header says more bytes follow than exist) is None,
    # not an IndexError; non-jpeg and a DQT-only prefix are None too
    b = encode_rgb(_red(8, 12))
    i = b.index(b"\xff\xc0") if b"\xff\xc0" in b else None
    for cut in (b"\x89PNG\r\n", b"\xff\xd8\xff\xdb\x00", b"\x00\x01"):
        assert jpeg_hw(cut) is None
    # truncate the jpeg at every point: never raises, returns (8, 12) or None
    for n in range(0, len(b) + 1):
        jpeg_hw(b[:n])
    assert jpeg_hw(b) == (8, 12)


def test_normalize_kept_frames_are_not_decoded():
    # #45: a q<=95 frame is kept verbatim without a full decode (the SOF header
    # suffices for the size check) — the bytes are byte-identical
    ok, buf = cv2.imencode(".jpg", _red(8, 12), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    b = bytes(buf.tobytes())
    _, _, (out,) = normalize_rgb([b])
    assert out == b                                 # verbatim, not re-encoded
    # a kept frame + a same-size over-95 frame: the second is the only decode
    ok2, buf2 = cv2.imencode(".jpg", _red(8, 12), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    _, _, out = normalize_rgb([b, bytes(buf2.tobytes())])
    assert out[0] == b and jpeg_quality(out[1]) == JPG_QUALITY
    # different sizes are still a size mismatch, kept or not
    ok3, buf3 = cv2.imencode(".jpg", _red(4, 4), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    with pytest.raises(ValueError, match="size"):
        normalize_rgb([b, bytes(buf3.tobytes())])


def test_normalize_rgb_array():
    H, W, frames = normalize_rgb(_red(8, 12))
    assert (H, W) == (8, 12) and len(frames) == 1
    assert jpeg_quality(frames[0]) == JPG_QUALITY


def test_normalize_rgb_paths(tmp_path):
    p = str(tmp_path / "f.png")
    cv2.imwrite(p, cv2.cvtColor(_red(8, 12), cv2.COLOR_RGB2BGR))
    H, W, frames = normalize_rgb([p])
    assert (H, W) == (8, 12) and len(frames) == 1


def test_normalize_rgb_mismatched_sizes_rejected():
    with pytest.raises(ValueError, match="size"):
        normalize_rgb([encode_rgb(_red(8, 12)), encode_rgb(_red(4, 4))])


def test_normalize_rgb_empty_is_a_clear_error():
    with pytest.raises(ValueError, match="empty"):
        normalize_rgb([])
