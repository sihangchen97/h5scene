"""Scan vggt_research/dataset h5 files for multi-camera (N>1) content and
per-camera resolution/intrinsic consistency — feasibility probe for h5camrig.

Reads each h5 read-only; decodes one frame per camera (n) to check the actual
stored H/W, compares depth shapes, and compares K[0,n] across cameras.
Outputs a per-file table + a summary. No writes to the datasets.
"""
import glob
import os
import sys
from collections import Counter

import cv2
import h5py
import numpy as np

ROOT = "/vol/juneleung-solar/lizard/vggt_research/dataset"
SKIP = ("_raw", "_preprocess", "_scripts", "_tests", ".bak",
        "pre_t", "pre_e")


def is_bad(path):
    return any(s in os.sep + path + os.sep for s in SKIP) or ".pre_" in path


def probe(path):
    out = {"path": path.rsplit(os.sep + "dataset" + os.sep, 1)[-1]}
    with h5py.File(path, "r") as h:
        # our format: attrs.TNHW — a non-rig file (no TNHW) is an error row,
        # there is no legacy layout left to scan
        T, N, H, W = (int(x) for x in h.attrs["TNHW"])
        out.update(T=T, N=N)
        # per-camera decoded size
        sizes, ks = [], []
        for n in range(N):
            img = cv2.imdecode(np.frombuffer(h["img"][f"0_{n}"][:], np.uint8),
                               cv2.IMREAD_COLOR)
            sizes.append(img.shape[:2] if img is not None else None)
        if "K" in h:
            for n in range(N):
                ks.append(h["K"][0, n][:].copy())
        out["cam_sizes"] = sizes
        out["hw_uniform"] = len(set(sizes)) <= 1
        if ks:
            k0 = ks[0]
            out["k_uniform"] = bool(all(np.array_equal(k0, k) for k in ks))
            out["k_sample"] = [float(k[0, 0]) for k in ks]
        out["has_depth"] = "depth" in h
        if out["has_depth"]:
            out["depth_shape"] = tuple(h["depth"].shape)
    return out


def main(limit=None):
    files = sorted(glob.glob(f"{ROOT}/**/*.h5", recursive=True))
    files = [f for f in files if not is_bad(f) and not f.endswith(".bak")]
    if limit:
        files = files[:limit]
    print(f"scanning {len(files)} h5 files (root={ROOT})")
    rows, errs = [], 0
    for i, p in enumerate(files):
        try:
            rows.append(probe(p))
        except Exception as e:
            errs += 1
            print(f"  ERR {p.rsplit('dataset/', 1)[-1]}: {e}")
        if i % 50 == 0:
            print(f"  …{i}/{len(files)}", flush=True)
    # summary
    ncnt = Counter(r.get("N") for r in rows)
    multi = [r for r in rows if r.get("N", 1) > 1]
    print("\n=== SUMMARY ===")
    print(f"files: {len(rows)}  (errors {errs})")
    print(f"N distribution: {dict(sorted(ncnt.items()))}")
    if multi:
        print(f"\nN>1 files ({len(multi)}):")
        for r in multi:
            print(f"  {r['path']}: N={r['N']} cam_sizes={r['cam_sizes']} "
                  f"hw_uniform={r.get('hw_uniform')} "
                  f"k_uniform={r.get('k_uniform')} k_fx={r.get('k_sample')} "
                  f"depth={r.get('depth_shape')}")
        bad_hw = [r for r in multi if not r.get("hw_uniform")]
        bad_k = [r for r in multi if r.get("k_uniform") is False]
        print(f"\n-> per-camera HW differs: {len(bad_hw)} / {len(multi)}")
        print(f"-> per-camera K differs:   {len(bad_k)} / {len(multi)}")
    # single-camera resolution diversity (h5cam: is (H,W) stable across files per dataset?)
    byds = {}
    for r in rows:
        if r.get("N", 1) == 1 and r.get("cam_sizes"):
            ds = r["path"].split(os.sep)[0]
            byds.setdefault(ds, Counter())[tuple(r["cam_sizes"][0])] += 1
    diverse = {k: dict(v) for k, v in byds.items() if len(v) > 1}
    print(f"\nper-dataset resolution diversity (N=1 files): {len(diverse)}/{len(byds)} datasets have >1 (H,W)")
    for k, v in sorted(diverse.items())[:12]:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)
