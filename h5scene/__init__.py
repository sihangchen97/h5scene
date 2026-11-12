"""h5scene: per-scene HDF5 writer + H5Scene file handle, unified (T, N, H, W) layout.

T = time axis (T=1: time-constant scene, e.g. static multi-view),
N = camera axis. Sections:

    <scene>.h5
      /rgb/{t}_{n}  uint8 1-D, image BYTES (jpg q<=95) — one dataset per frame,
                     key derived from (t, n); T*N keys total
      /depth        uint16 (T,N,h,w) mm, h<=H w<=W, aspect ratio H/W matches,
                    0 AND 65535 both invalid, lossless gzip4 + shuffle;
                    depth(t,n) resizes (h,w) -> (H,W) on read (nearest)  [optional]
      /mask         bool   (T,N,H,W) 1 = valid pixel, 0 = padding, gzip4 [optional]
      /K            f32    (T,N,3,3) "TN33"; (3,3) input broadcasts     [optional]
      /T            f32    (T,N,4,4) "TN44"; (4,4) input broadcasts     [optional]
      /depth_conf   f16    (T,N,h,w) 0-1 (1 = full confidence), gzip4 —
                    same (h,w) as /depth, resized -> (H,W) on read      [optional]
      derived: s.depth_mask(t, n, conf_min=…) — (H,W) bool, not stored:
                 depth not an invalid sentinel (0 AND 65535 excluded) and,
                 if depth_conf present, conf > (0 if conf_min is None, else conf_min)
      /meta/{dataset,scene,seq,suffix,split,meta_json}
              utf-8 strings (dataset-side context; meta_json = source verbatim).
              4 location keys: (dataset, scene[, seq]) shared by several h5 =
              the same scene's multi-view cameras; suffix marks variants (train/val,
              camera/version) — see docs/meta_key_decisions.md
      /meta/frame_indices  int64 (T,) source frame numbers; only when
              != arange(T) (dense: t already is the source frame)       [optional]
      /meta/tags/*         utf-8 strings (data_type/dynamics/…)        [optional]
      attrs: TNHW = (T, N, H, W) int64 array, is_metric bool (data-side;
              legacy files still read /meta/is_metric)

rgb input: a (T,N,H,W,3) or (H,W,3) uint8 array, a per-frame image-BYTES
sequence, or file paths (flat: T frames N=1; (T,N): N cameras per frame).
All stored as jpg with quality <= 95: arrays are encoded to jpg q95; existing
bytes/paths are stored verbatim when their detected cv2/libjpeg quality is
<= 95, and re-encoded to jpg q95 otherwise (quality > 95, or a different
encoding such as png — see rgb._load_rgb_frames).

Depth input rule: float = meters, uint16 = already mm. Either way the writer
clips d > 65.535 m to 0 (invalid) — never saturated to 65535, so both 0 and
65535 mean invalid and a read-time `>= 65535 -> 0` double-safety stays valid.
Depth (and depth_conf) may be stored at a smaller (h, w) than the scene (H, W)
— the aspect ratio H/W must match (h*W == w*H) and a present depth_conf must
have the SAME (h, w); readers resize (h, w) back to (H, W) with INTER_NEAREST.

depth_conf input: 0-1 floats (clipped) or 0-65535 uint16; stored fp16 0-1.

mask input: bool/uint8, 1 = valid pixel, 0 = padding, FULL scene (H, W) only —
(T,N,H,W) per pixel, (H,W) broadcast to every (t,n), or (T,N) per slot
(whole frame is padding).

K/T input shapes: (3,3)/(4,4) broadcast to every (t,n); (T,N,…) stored as-is.

H5Scene is a direct file handle (no in-memory data):

    s = H5Scene(path, T, N, H, W)     # create (overwrites), TNHW pinned in attrs
    s.set_rgb(frames_or_array)        # writes /rgb directly (shape validated)
    s.set_depth(d); s.set_mask(m); s.set_K(K); s.set_pose(P); s.set_depth_conf(c)
    s.set_meta(dataset=…, scene=…, seq=…, suffix=…, meta_json=…)
    s.close()                         # (or: with H5Scene(…) as s: …)

    s = H5Scene(path)                 # open an existing file (TNHW read from attrs)
    s.shape, s.keys, s.has(name)
    s.rgb_bytes(t, n)  /  s.rgb(t, n)          # stored bytes / decoded (H,W,3)
    s.depth(t, n, meters=…) / s.K(t, n) / s.pose(t, n) / s.mask(t, n) / s.depth_conf(t, n)
    s.dataset, s.scene, s.seq, s.suffix, s.split, s.meta_json, s.is_metric
    s.frame_indices, s.tags
"""
from .scene import H5Scene

__all__ = ["H5Scene"]
