"""Shared HDF5 section IO: the shared section encode/decode helpers + presence (has) that
both H5Cam (single-camera, rank (T, …)) and H5CamRig (multi-camera, rank
(T, N, …)) build their set_*/read methods on. Rank-agnostic by design: the
READ helpers operate on a single frame array (the caller passes
h['depth'][t] or h['depth'][t, n]), and the WRITE helpers take an explicit
(T, N) plus the (T, N) key convention "{t}_{n}"; H5Cam collapses the N axis
itself. So the two classes share these without a common parent.

Encoding / layout policy (see README + docs/meta_key_decisions.md):
- /rgb: image bytes, jpg q<=95 (rgb._normalize_rgb / _load_rgb_frames); no
  bytes-level reader — decode via cv2 only.
- /depth: uint16 mm; 0 AND 65535 both invalid. Writer clips d > 65.535 m ->
  0 (never saturates to 65535); reader re-clips >= 65535 -> 0 (double
  safety). May be stored smaller (h, w) than the scene (H, W) only if the
  aspect ratio H/W matches (h*W == w*H); reads resize (h, w) back to (H, W)
  with INTER_NEAREST (sentinels never interpolated).
- /mask: bool, gzip4 (1 = valid, 0 = padding); measured ~180x smaller than raw
  on a mostly-valid probe, no shuffle.
- /K f32 / /T f32: (3,3)/(4,4) broadcast; stored RAW (no gzip) — measured
  bigger with gzip for short sequences (K break-even ~106 frames, pose ~64).
- /depth_conf: f16 0-1, SAME (h, w) as /depth, resized -> (H, W) on read.
- /meta: dataset/scene/seq/suffix/split/meta_json (utf-8) + frame_indices
  (only when != arange(T)) + tags/*; the 4 location keys (dataset, scene[,
  seq]) shared by several h5 = the same scene's multi-view cameras.
"""
import cv2
import h5py
import numpy as np

from .convert import (_broadcast, _depth_hw_ok, _frame_keys, _resize_hw,
                      _to_depth_conf, _to_depth_mm, _to_mask)
from .rgb import _normalize_rgb

_STR = h5py.string_dtype(encoding="utf-8")


# ---- low-level dataset IO (rank-agnostic) ----
def write_section(g, name, data, **kw):
    """Replace /<name> with data (del first if present). Low-level write."""
    if name in g:
        del g[name]
    g.create_dataset(name, data=data, **kw)


def write_rgb_frame(g, key, b):
    """Store one image-bytes frame at /rgb/{key} (uint8 1-D)."""
    rg = g.require_group("rgb")
    rg.create_dataset(key, data=np.frombuffer(bytes(b), np.uint8))


def decode_rgb(b):
    """Image bytes -> (H, W, 3) uint8 RGB (cv2). Raises if not decodable."""
    img = cv2.imdecode(b, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("rgb frame is not a decodable image")
    return img


# ---- rgb encode (jpg q<=95 policy) — (T, N) "{t}_{n}" convention ----
def write_rgb(g, rgb, T, N):
    """rgb -> /rgb/{t}_{n} (replaces all keys). Input: a (T,N,H,W,3)/(H,W,3)
    uint8 array / an image-bytes sequence / file paths; every frame stored as
    jpg q<=95 per rgb._normalize_rgb. Shape validated against (T, N)."""
    t, n, h, w, frames = _normalize_rgb(rgb)
    if (t, n) != (T, N):
        raise ValueError(f"rgb shape (T,N) = ({t},{n}) != scene ({T},{N})")
    rg = g.require_group("rgb")
    for k in list(rg):
        del rg[k]
    for fid, b in zip(_frame_keys(T, N), frames):
        rg.create_dataset(fid, data=np.frombuffer(b, np.uint8))


# ---- depth / depth_conf / mask / K / pose: (T, N) encode + write ----
def _check_depth_hw(g, name, h, w, H, W):
    """(h, w) may be < (H, W) only if aspect ratio H/W matches (h*W == w*H);
    a present sibling (depth<->depth_conf) must be stored at the same (h, w)."""
    if not _depth_hw_ok(h, w, H, W):
        raise ValueError(f"{name} size ({h},{w}) incompatible with scene "
                         f"(H,W) = ({H},{W}): smaller but aspect ratio H/W must match")
    other = {"depth": "depth_conf", "depth_conf": "depth"}[name]
    if other in g:
        s = g[other].shape
        if (s[-2], s[-1]) != (h, w):
            raise ValueError(f"{name} size ({h},{w}) != {other} size ({s[-2]},{s[-1]})")


def write_depth(g, depth, T, N, H, W):
    """float meters (or uint16 mm); > 65.535 m -> 0 (invalid); may be stored
    smaller (h, w) if aspect ratio matches (and == depth_conf's (h, w)).
    Stored lossless gzip4 + shuffle."""
    d = _to_depth_mm(depth, T, N)
    _check_depth_hw(g, "depth", d.shape[-2], d.shape[-1], H, W)
    write_section(g, "depth", d, chunks=(1, 1, d.shape[-2], d.shape[-1]),
                  compression="gzip", compression_opts=4, shuffle=True)


def write_depth_conf(g, conf, T, N, H, W):
    """0-1 floats (or 0-65535 uint16); stored f16 0-1, gzip4 (no shuffle:
    most values 1.0 -> shuffle measured 3% larger). Same (h, w) as depth."""
    c = _to_depth_conf(conf, T, N)
    _check_depth_hw(g, "depth_conf", c.shape[-2], c.shape[-1], H, W)
    write_section(g, "depth_conf", c, chunks=(1, 1, c.shape[-2], c.shape[-1]),
                  compression="gzip", compression_opts=4)


def write_mask(g, mask, T, N, H, W):
    """bool/uint8: (T,N,H,W) pixels / (H,W) broadcast / (T,N) slots; full
    (H, W) only, stored gzip4."""
    write_section(g, "mask", _to_mask(mask, T, N, H, W),
                  chunks=(1, 1, H, W), compression="gzip", compression_opts=4)


def write_K(g, K, T, N):
    """(3,3) broadcast or (T,N,3,3) float; stored raw (no gzip)."""
    try:
        write_section(g, "K", _broadcast(K, np.float32, T, N, (3, 3)))
    except ValueError as e:
        raise ValueError(f"K: {e}") from None


def write_pose(g, pose, T, N):
    """(4,4) broadcast or (T,N,4,4) float c2w; stored raw (no gzip)."""
    try:
        write_section(g, "T", _broadcast(pose, np.float32, T, N, (4, 4)))
    except ValueError as e:
        raise ValueError(f"pose: {e}") from None


# ---- depth / depth_conf / mask: READ (single frame -> (H, W)) ----
def depth_to_mm(frame, H, W):
    """One (h, w) depth frame -> (H, W) uint16 mm; a (h, w) < (H, W) frame is
    resized (nearest) and sentinels re-clipped (>= 65535 -> 0)."""
    d = _resize_hw(frame, H, W, cv2.INTER_NEAREST).astype(np.int32)
    d[d >= 65535] = 0
    return d.clip(0, 65535).astype(np.uint16)


def depth_to_meters(frame, H, W):
    """One (h, w) depth frame -> (H, W) float32 meters."""
    return (depth_to_mm(frame, H, W).astype(np.int32) / 1000.0).astype(np.float32)


def depth_conf_to_f32(frame, H, W):
    """One (h, w) confidence frame -> (H, W) float32 0-1 (resized, nearest)."""
    return _resize_hw(frame, H, W, cv2.INTER_NEAREST).astype(np.float32)


def depth_mask(depth_mm, conf_f32, conf_min=None):
    """depth_mm / conf_f32 both (H, W) -> (H, W) bool: depth not an invalid
    sentinel (0 AND 65535 excluded) and, if conf is present, conf > (0 when
    conf_min is None, else conf_min)."""
    m = (depth_mm != 0) & (depth_mm != 65535)
    if conf_f32 is not None:
        m &= (conf_f32 > 0.0) if conf_min is None else (conf_f32 > conf_min)
    return m


# ---- meta + presence (rank-agnostic, operate on the file group g) ----
def has(g, name):
    """Presence of a section / meta key in file group g."""
    m = g["meta"]
    return {"rgb": len(g["rgb"]) > 0, "depth": "depth" in g, "mask": "mask" in g,
            "K": "K" in g, "T": "T" in g, "depth_conf": "depth_conf" in g,
            "is_metric": "is_metric" in g.attrs or "is_metric" in m,
            "dataset": "dataset" in m, "scene": "scene" in m,
            "seq": "seq" in m, "suffix": "suffix" in m, "split": "split" in m,
            "meta_json": "meta_json" in m or "meta_json" in g,
            "frame_indices": "frame_indices" in m, "tags": "tags" in m}[name]


def tags_dict(g):
    """dict[str, str] from /meta/tags/, or None when absent."""
    m = g["meta"]
    if "tags" not in m:
        return None
    return {k: m["tags"][k][()].decode() for k in m["tags"]}


def is_metric(g):
    """bool; attrs (data-side) preferred, legacy /meta/is_metric as fallback."""
    if "is_metric" in g.attrs:
        return bool(g.attrs["is_metric"])
    return bool(g["meta"]["is_metric"][()])
