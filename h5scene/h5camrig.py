"""
H5CamRig: N H5Cam files -> one multi-camera reference file (T, N, Hmax, Wmax) - REFERENCES only, nothing copied into the rig file.

- STRICT checks: same T, same is_metric, same section set, rgb per-frame
  size == declared (H, W), depth/depth_conf/mask per-frame size == frame 0,
  and H or W identical across ALL cameras (the other axis = max; both axes
  differ -> rejected).
- /rgb: ExternalLinks to each camera own /rgb/{t} (0 bytes, no copy, no
  re-encode); rgb(t, n) decodes at camera n NATIVE (H, W).
- /depth /mask /K /T /depth_conf: VIRTUAL datasets referencing the camera
  files (0 bytes on disk), at (T, N, Hmax, Wmax); smaller cameras map
  per-frame (h, w) blocks to (t, n, :h, :w) with fillvalue pad.
- attrs.cam_files (absolute) + attrs.cam_hw; H5CamRig(open) verifies the
  camera files exist, because a missing/renamed source makes VDS reads
  SILENTLY return the fillvalue (ExternalLinks raise KeyError).
- the result is also a plain H5Base file, openable directly with H5Base.
"""

import os

import cv2
import h5py
import numpy as np

from .convert import _depth_hw_ok
from .h5cam import H5Cam
from .h5base import H5Base

_SECTIONS = (("depth", np.uint16, 0, (2,)),
             ("mask", np.bool_, False, (2,)),
             ("K", np.float32, 0.0, (3, 3)),
             ("T", np.float32, 0.0, (4, 4)),
             ("depth_conf", np.float16, 0.0, (2,)))


class H5CamRig:
    """N H5Cam files -> one multi-camera file (T, N, Hmax, Wmax): /rgb
    physically copied (jpg variable-length, not virtualizable), the
    fixed-shape sections /depth /mask /K /T /depth_conf as HDF5 VIRTUAL
    datasets referencing the camera files (0 bytes on disk; compressed
    source data mapped directly; pad regions covered by fillvalue).

    Build:    H5CamRig(out_path, cam_paths, meta={...})
    Open:     H5CamRig(out_path)   # verifies attrs.cam_files exist, then H5CamRig(out_path).scene
    """

    def __init__(self, out_path, cam_paths=None, meta=None):
        self.path = str(out_path)
        if cam_paths is None:            # open
            self._h = h5py.File(self.path, "r")
            if "TNHW" not in self._h.attrs or "cam_files" not in self._h.attrs:
                self.close()
                raise ValueError(f"{self.path}: not an H5CamRig file (no attrs.TNHW / cam_files)")
            missing = [p for p in self._h.attrs["cam_files"] if not os.path.exists(p)]
            if missing:
                self.close()
                raise FileNotFoundError(
                    f"{self.path}: missing camera source file(s) — VDS would "
                    f"silently read fillvalues: {missing}")
            self.scene = H5Base(self.path)
            self._h.close()
            self._h = self.scene._h
            return
        self._build(cam_paths, meta)

    @property
    def cam_files(self):
        """List[str] of the source camera files (absolute) this rig references."""
        return [str(p) for p in self.scene._h.attrs["cam_files"]]

    @property
    def cam_hw(self):
        """List[(H, W)] of each camera's native resolution, in cam order."""
        return [tuple(int(x) for x in row) for row in self.scene._h.attrs["cam_hw"]]

    def close(self):
        if self._h is not None:
            if "scene" in self.__dict__:
                self.scene.close()
            else:
                self._h.close()
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- build ----
    def _build(self, cam_paths, meta):
        cams = [H5Cam(p) for p in cam_paths]
        try:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
            tsets = {c.T for c in cams}
            if len(tsets) != 1:
                raise ValueError(f"camera T mismatch: {sorted(tsets)}")
            T = tsets.pop()
            mets = {c.is_metric for c in cams}
            if len(mets) != 1:
                raise ValueError(f"camera is_metric mismatch: {sorted(mets)}")
            is_metric = mets.pop()
            hs = {c.H for c in cams}
            ws = {c.W for c in cams}
            if len(hs) > 1 and len(ws) > 1:
                raise ValueError(f"rig needs H or W identical across cameras; "
                                 f"got H={sorted(hs)} W={sorted(ws)}")
            H, W, N = max(c.H for c in cams), max(c.W for c in cams), len(cams)
            secs = [tuple(k for k, *_ in _SECTIONS if k in c._h) for c in cams]
            if len(set(secs)) != 1:
                raise ValueError(f"camera section sets differ: {secs}")
            secs = secs[0]
            # strict per-frame size checks
            for n, c in enumerate(cams):
                for t in range(T):
                    img = cv2.imdecode(c._h["rgb"][f"{t}"][:], cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError(f"cam {n} rgb frame {t} not decodable")
                    if img.shape[:2] != (c.H, c.W):
                        raise ValueError(f"cam {n} rgb frame {t} size {img.shape[:2]} "
                                         f"!= declared ({c.H},{c.W})")
                for name, *_ in _SECTIONS:
                    if name in secs and name not in ("K", "T"):
                        s0 = c._h[name][0].shape
                        for t in range(1, T):
                            if c._h[name][t].shape != s0:
                                raise ValueError(f"cam {n} {name} frame {t} shape {c._h[name][t].shape} "
                                                 f"!= frame 0 {s0}")

            with h5py.File(self.path, "w", libver="latest") as h:
                h.attrs["TNHW"] = np.array((T, N, H, W), np.int64)
                h.attrs["is_metric"] = is_metric
                abs_paths = [os.path.abspath(c.path) for c in cams]
                h.attrs["cam_files"] = np.array(abs_paths, dtype=_STR)
                h.attrs["cam_hw"] = np.array([(c.H, c.W) for c in cams], np.int64)
                # /rgb: nothing stored — each (t, n) is an ExternalLink to the
                # camera's own /rgb/{t} (variable-length jpg cannot be a virtual
                # dataset). rgb(t, n) therefore decodes at camera n's NATIVE
                # (H, W); the pad-to-common-HW only happens for the VDS sections.
                rg = h.create_group("rgb")
                for n, c in enumerate(cams):
                    for t in range(T):
                        rg[f"{t}_{n}"] = h5py.ExternalLink(abs_paths[n], f"/rgb/{t}")
                # fixed-shape sections: virtual datasets referencing the cameras
                for name, dtype, fill, rank in _SECTIONS:
                    if name not in secs:
                        continue
                    shp = (H, W) if rank == (2,) else rank
                    lay = h5py.VirtualLayout(shape=(T, N, *shp), dtype=dtype)
                    for n, c in enumerate(cams):
                        if name in ("K", "T"):        # (3,3)/(4,4) frames: one block per camera
                            lay[:, n, ...] = h5py.VirtualSource(
                                os.path.abspath(c.path), name, shape=(T, *c._h[name][0].shape))
                        elif c._h[name][0].shape == (H, W) and (c.H, c.W) == (H, W):
                            lay[:, n, ...] = h5py.VirtualSource(        # full-size frame: one block
                                os.path.abspath(c.path), name, shape=(T, H, W))
                        else:  # per-frame (h, w) blocks; pad region -> fillvalue
                            for t in range(T):
                                h_, w_ = c._h[name][t].shape
                                lay[t, n, :h_, :w_] = h5py.VirtualSource(
                                    os.path.abspath(c.path), name, shape=(h_, w_))
                    h.create_virtual_dataset(name, lay, fillvalue=fill)
                m = h.require_group("meta")       # H5Base requires /meta on open
                for k, v in (meta or {}).items():
                    if v is not None:
                        m.create_dataset(k, data=np.array(str(v), dtype=_STR))
        finally:
            for c in cams:
                c.close()
        self._h = h5py.File(self.path, "r")
        self.scene = H5Base(self.path)
        self._h.close()
        self._h = self.scene._h

    # ---- passthrough reads: rig.shape / rig.rgb(t, n) / rig.depth(...) / rig.has / rig.tags …
    # (everything on H5Base, incl. the .shape property) ----
    def __getattr__(self, name):
        if name.startswith("_") or name in ("path", "scene"):
            raise AttributeError(name)
        try:
            sc = object.__getattribute__(self, "scene")
        except AttributeError:
            raise AttributeError(name)
        return getattr(sc, name)


_STR = h5py.string_dtype()
