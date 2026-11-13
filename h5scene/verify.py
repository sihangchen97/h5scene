"""h5scene.verify — the post-conversion check run (eval request E6):

    python -m h5scene.verify <file.h5 | dir> [--json out.json]

Per H5Cam file it reports: opens OK; (raw depth == 65535).sum() (must be 0 —
the API maps >= 65.535 m to 0, so a stored 65535 means foreign/corrupt
data); the valid-depth min/max; the K/pose convention checks (E3); every img
frame decodes at the declared (H, W); the src_frame_ids uniqueness. The exit
code is 1 when any check fails. A directory is scanned for *.h5 (non-recursive).
"""
import argparse
import json
import os
import os.path as osp
import sys

import h5py
import numpy as np

from .codecs import SECTIONS, check_K, check_pose
from .codecs.rgb import decode_rgb
from .h5cam import H5Cam
from .h5camrig import H5CamRig
from .h5geo import H5Geo

# the K/pose check tolerances (the same rules as set_K / set_pose_c2w)
_EPS = 1e-6
_U16_MAX = int(np.iinfo(np.uint16).max)


def _check_K_rows(krows):
    errs = []
    for i, m in enumerate(krows):
        fx, fy = m[0, 0], m[1, 1]
        if not (fx > 0 and fy > 0):
            errs.append(f"K frame {i}: fx / fy must be > 0, got {fx} / {fy}")
        if abs(m[2, 2] - 1.0) > _EPS:
            errs.append(f"K frame {i}: K[2,2] must be 1 (pinhole), got {m[2, 2]}")
        for r, c, name in ((1, 0, "skew K[1,0]"), (2, 1, "K[2,1]"), (2, 0, "K[2,0]")):
            if abs(m[r, c]) > _EPS:
                errs.append(f"K frame {i}: {name} must be 0, got {m[r, c]}")
    return errs


def _check_pose_rows(prows):
    errs = []
    for i, p in enumerate(prows):
        if not np.allclose(p[3], [0, 0, 0, 1], atol=_EPS):
            errs.append(f"pose frame {i}: last row must be [0, 0, 0, 1], got {p[3]}")
        R = p[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
            errs.append(f"pose frame {i}: R must be orthonormal")
        if abs(np.linalg.det(R) - 1.0) > 1e-3:
            errs.append(f"pose frame {i}: det(R) must be 1, got {np.linalg.det(R)}")
    return errs


def verify_file(path):
    """One H5Cam file -> {"path", "ok", "checks" {name: {...}}}. A file that
    is not an H5Cam (or does not exist) is a hard failure. NOT a light read:
    the depth/img checks load the full data (H5Cam.peek is the light one)."""
    out = {"path": str(path), "ok": False, "checks": {}}
    if not osp.exists(path):
        out["checks"]["open"] = {"ok": False, "error": "file missing"}
        return out
    try:
        cam = H5Cam.open(path)
    except Exception as e:
        out["checks"]["open"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return out
    out["checks"]["open"] = {"ok": True}
    T, H, W = cam.shape
    with cam:
        if cam.has("depth"):
            d = cam._h["depth"][:]
            valid = (d != 0) & (d != _U16_MAX)
            n65535 = int((d == _U16_MAX).sum())
            out["checks"]["depth"] = {
                "ok": n65535 == 0, "n_65535": n65535,
                "valid_min": int(d[valid].min()) if valid.any() else None,
                "valid_max": int(d[valid].max()) if valid.any() else None,
                "valid_frac": round(float(valid.mean()), 6)}
        if cam.has("K"):
            errs = _check_K_rows(cam._h["K"][:])
            out["checks"]["K"] = {"ok": not errs, "errors": errs}
        if cam.has("T"):
            errs = _check_pose_rows(cam._h["T"][:])
            out["checks"]["pose"] = {"ok": not errs, "errors": errs}
        if cam.has("img"):
            bad = []
            for t in range(T):
                try:
                    img = decode_rgb(cam.img_bytes(t))
                    if img.shape[:2] != (H, W):
                        bad.append((t, f"decoded {img.shape[:2]} != declared ({H},{W})"))
                except Exception as e:
                    bad.append((t, f"{type(e).__name__}: {e}"))
            out["checks"]["img"] = {"ok": not bad, "bad_frames": bad,
                                    "n": T, "hw": [H, W]}
        ids = cam.src_frame_ids
        if ids is not None:
            from collections import Counter
            dup = sorted(x for x, n in Counter(ids).items() if n > 1)
            out["checks"]["src_frame_ids"] = {
                "ok": not dup and len(ids) == T, "n": len(ids), "duplicates": dup}
    out["ok"] = all(c["ok"] for c in out["checks"].values())
    return out


def _verify_other(path, ft):
    """An H5CamRig / H5Geo file: verified by open only (their checks live in
    the constructors)."""
    try:
        obj = H5CamRig.open(path) if ft == "h5camrig" else H5Geo.open(path)
        obj.close()
        return {"path": path, "ok": True, "file_type": ft,
                "checks": {"open": {"ok": True}}}
    except Exception as e:
        return {"path": path, "ok": False, "file_type": ft,
                "checks": {"open": {"ok": False,
                                    "error": f"{type(e).__name__}: {e}"}}}


def verify_dir(d):
    """Every *.h5 in d (non-recursive), sorted."""
    results = []
    for f in sorted(os.listdir(d)):
        p = osp.join(d, f)
        if not f.endswith(".h5") or not osp.isfile(p):
            continue
        try:
            with h5py.File(p, "r") as h:
                ft = str(h.attrs.get("file_type", ""))
        except Exception as e:
            results.append({"path": p, "ok": False, "checks": {
                "open": {"ok": False, "error": f"{type(e).__name__}: {e}"}}})
            continue
        results.append(verify_file(p) if ft == "h5cam" else _verify_other(p, ft))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m h5scene.verify",
                                 description="check h5scene conversion files")
    ap.add_argument("paths", nargs="+",
                    help="H5Cam files or directories (non-recursive)")
    ap.add_argument("--json", metavar="OUT", help="write the results as JSON")
    args = ap.parse_args(argv)
    results = []
    for p in args.paths:
        if osp.isdir(p):
            results.extend(verify_dir(p))
        elif osp.exists(p):
            results.append(verify_file(p))
        else:
            results.append({"path": p, "ok": False, "checks": {
                "open": {"ok": False, "error": "file missing"}}})
    for r in results:
        mark = "OK  " if r["ok"] else "FAIL"
        detail = "; ".join(
            f"{k}: {v.get('error') or (v.get('errors') or v.get('bad_frames') or v.get('duplicates') or '')}"
            for k, v in r["checks"].items() if not v["ok"])
        line = f"{mark} {r['path']}"
        if r.get("file_type"):
            line += f"  [{r['file_type']}]"
        if detail:
            line += f"  [{detail}]"
        print(line)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"-> {args.json}")
    return 0 if results and all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
