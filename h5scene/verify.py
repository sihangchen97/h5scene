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

from .codecs import K_row_errors, pose_row_errors
from .codecs.rgb import decode_rgb
from .h5cam import H5Cam
from .h5camrig import H5CamRig
from .h5geo import H5Geo

# the invalid-sentinel value (0 and 65535 are both "invalid")
_U16_MAX = int(np.iinfo(np.uint16).max)


def verify_file(path):
    """One H5Cam file -> {"path", "ok", "checks" {name: {...}}}. A file that
    is not an H5Cam (or does not exist) is a hard failure. NOT a light read:
    it reads every depth frame (one at a time) and every img frame
    (H5Cam.peek is the light one)."""
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
            # one frame at a time (a long sequence must not need T * h * w in RAM)
            d = cam._h["depth"]
            n65535 = nv = 0
            vmin, vmax = None, None
            for t in range(T):
                f = d[t]
                v = (f != 0) & (f != _U16_MAX)
                n65535 += int((f == _U16_MAX).sum())
                if v.any():
                    nv += int(v.sum())
                    lo, hi = int(f[v].min()), int(f[v].max())
                    vmin = lo if vmin is None else min(vmin, lo)
                    vmax = hi if vmax is None else max(vmax, hi)
            n = T * d.shape[1] * d.shape[2]
            out["checks"]["depth"] = {
                "ok": n65535 == 0, "n_65535": n65535,
                "valid_min": vmin, "valid_max": vmax,
                "valid_frac": round(nv / n, 6)}
        if cam.has("K"):
            errs = K_row_errors(cam._h["K"][:].astype(np.float64))
            out["checks"]["K"] = {"ok": not errs, "errors": errs}
        if cam.has("T"):
            errs = pose_row_errors(cam._h["T"][:].astype(np.float64))
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
        if f.endswith(".h5") and osp.isfile(p):
            results.append(verify_any(p))
    return results


def verify_any(path):
    """One EXISTING h5scene file, dispatched on attrs.file_type (a single
    file argument must verify as its own class, not as an H5Cam)."""
    try:
        with h5py.File(str(path), "r") as h:
            ft = str(h.attrs.get("file_type", ""))
    except Exception as e:
        return {"path": str(path), "ok": False, "checks": {"open": {
            "ok": False, "error": f"{type(e).__name__}: {e}"}}}
    if ft == "h5cam":
        return verify_file(path)
    if ft in ("h5camrig", "h5geo"):
        return _verify_other(path, ft)
    return {"path": str(path), "ok": False, "checks": {"open": {
        "ok": False, "error": f"not an h5scene file (attrs.file_type = {ft!r})"}}}


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
            results.append(verify_any(p))
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
