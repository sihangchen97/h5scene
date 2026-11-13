"""Compression sweep for the H5Geo lossless attribute policy.

Runs the option matrix from requirements_gt_geometry/IMPLEMENTATION_SPEC.md
§5.0/§5.2 (filters × chunk rows × ptnum layout) over the sample plys
(default: _tobedelete/requirements_gt_geometry/sample_plys/ — the originals
moved there after the 2026-11-13 run), measures per cell:
stored bytes, ratio to raw, write time, full-read time (ds[:], the consumer's
only access pattern), and asserts a bit-exact round trip. Applies the §5.0
decision rule per attribute, quantifies the C7 per-source exception and the
C8 info-only options, projects the 43-scene total, and writes:
  - the report  -> tests/probes/geo/README.md
  - the 50k-point probes -> tests/probes/geo/<source>/*.npy  (§5.3)

The binary ply parser below is deliberately private to this script (D7: no
ply reader in the library). Supports binary_little_endian files with
element vertex (x/y/z float|double, red/green/blue uchar, [nx/ny/nz double])
and optional element face (property list uchar int, all-triangle).

Usage:
    pixi run smoke-geo [ply_dir] [--write-probes]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "tests" / "probes" / "geo"
REPORT = PROBE_DIR / "README.md"
MB = 1e6
PROBE_N = 50_000

# §5.0 options (C1–C4; lzf = C3), as (name, compression, level, shuffle)
OPTIONS = [
    ("none", None, None, False),
    ("none+sh", None, None, True),
    ("gzip1", "gzip", 1, False),
    ("gzip1+sh", "gzip", 1, True),
    ("gzip4", "gzip", 4, False),
    ("gzip4+sh", "gzip", 4, True),
    ("gzip6", "gzip", 6, False),
    ("gzip6+sh", "gzip", 6, True),
    ("gzip9", "gzip", 9, False),
    ("gzip9+sh", "gzip", 9, True),
    ("lzf", "lzf", None, False),
    ("lzf+sh", "lzf", None, True),
]
CHUNKS = (16_384, 65_536, 262_144)

# C1 per attribute: multi-byte gzip4+shuffle, 1-byte (Cd) plain gzip4
C1_NAME = {"P": "gzip4+sh", "N": "gzip4+sh", "ptnum": "gzip4+sh", "Cd": "gzip4"}

# per-source total ply bytes over all 43 scenes (~1.8 GB, REQUIREMENTS_GT_GEOMETRY.md):
# dtu 13 x 27 MB, nrgbd 8 x 27 MB, 7scenes 7 x 51 MB, hiroom 6 x 38 MB;
# scannetpp = the remainder of 1.8 GB (11 scenes, 14–142 MB each) — an estimate.
SRC_TOTAL_BYTES = {
    "dtu": 13 * 27e6,
    "nrgbd": 8 * 27e6,
    "7scenes": 7 * 51e6,
    "hiroom": 6 * 38e6,
    "scannetpp": 1.8e9 - (13 * 27e6 + 8 * 27e6 + 7 * 51e6 + 6 * 38e6),
}

_PLY_DT = {"float": "<f4", "double": "<f8", "uchar": "<u1", "int": "<i4"}


def source_of(name: str) -> str:
    for s in ("7scenes", "dtu", "hiroom", "nrgbd", "scannetpp"):
        if name.startswith(s):
            return s
    sys.exit(f"unknown source in {name}")


# ---- private binary-ply parser (D7: not in the library) -------------------

def read_ply(path: Path):
    """-> (P (M,3) f32, Cd (M,3) u8 | None, N (M,3) f32 | None, tri (F,3) i32 | None).
    Positions and normals are cast to f32 here, exactly as the H5Geo writers will."""
    with open(path, "rb") as f:
        props, nverts, nfaces, cur = {}, 0, 0, None
        face_prop = None
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            if not line or line.startswith(("comment", "obj_info", "format")):
                continue
            if line.startswith("element vertex"):
                nverts, cur, props = int(line.split()[2]), "vertex", {}
            elif line.startswith("element face"):
                nfaces, cur = int(line.split()[2]), "face"
            elif line.startswith("property") and cur == "vertex":
                p = line.split()
                props[p[2]] = _PLY_DT[p[1]]
            elif line.startswith("property") and cur == "face":
                face_prop = line.split()[1:]
        assert nverts and all(k in props for k in "xyz"), path
        assert face_prop is None or face_prop[:3] == ["list", "uchar", "int"], face_prop
        dt = np.dtype(list(props.items()))
        v = np.frombuffer(f.read(nverts * dt.itemsize), dtype=dt)
        P = np.ascontiguousarray(
            np.stack([v[k] for k in ("x", "y", "z")], axis=1).astype(np.float32))
        Cd = (np.ascontiguousarray(
            np.stack([v[k] for k in ("red", "green", "blue")], axis=1))
            if "red" in props else None)
        N = (np.ascontiguousarray(
            np.stack([v[k] for k in ("nx", "ny", "nz")], axis=1).astype(np.float32))
            if "nx" in props else None)
        tri = None
        if nfaces:
            # 13 bytes per face: 1-byte vertex count + 3 int32 indices
            blob = f.read()
            assert len(blob) == nfaces * 13, \
                f"face blob {len(blob)} != {nfaces} * 13"
            b = np.frombuffer(blob, dtype="<u1")
            assert (b[::13] == 3).all(), "non-triangle face in sample ply"
            keep = np.ones(b.shape, bool)
            keep[::13] = False                       # drop the count byte of each face
            tri = np.ascontiguousarray(b[keep].view("<i4").reshape(nfaces, 3))
    return P, Cd, N, tri


# ---- one sweep cell ---------------------------------------------------------

def bench(fpath: Path, arr: np.ndarray, comp, lvl, shfl, chunk) -> dict:
    """stored bytes / ratio / write s / read s (min of 3 warm reads) for one
    filter × chunk cell, with a bit-exact round-trip assert."""
    kw = {}
    if comp is not None:
        kw = {"compression": comp, "shuffle": shfl}
        if lvl is not None:
            kw["compression_opts"] = lvl
    elif shfl:
        kw = {"shuffle": True}
    chunks = (min(arr.shape[0], chunk), *arr.shape[1:])
    t0 = time.perf_counter()
    try:
        with h5py.File(fpath, "w") as f:
            d = f.create_dataset("data", data=arr, **kw, chunks=chunks)
            f.flush()
            stored = d.id.get_storage_size()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    t_w = time.perf_counter() - t0
    r_s, ok = float("inf"), False
    with h5py.File(fpath, "r") as f:
        d = f["data"]
        _ = d[:]                                          # warm the page cache
        for _ in range(3):
            t0 = time.perf_counter()
            ok = np.array_equal(d[:], arr)
            r_s = min(r_s, time.perf_counter() - t0)
    if not ok:
        return {"error": "ROUND-TRIP MISMATCH"}
    return {"stored": stored, "ratio": stored / arr.nbytes,
            "w_s": t_w, "r_s": r_s, "raw": arr.nbytes}


# ---- probes (§5.3) ----------------------------------------------------------

def write_probes(srcs: dict) -> dict:
    out = {}
    for s, a in srcs.items():
        out[s] = {}
        d = PROBE_DIR / s
        d.mkdir(parents=True, exist_ok=True)
        P, Cd, N, tri = a["P"], a["Cd"], a["N"], a["tri"]
        n = min(PROBE_N, len(P))
        np.save(d / "P.npy", P[:n])
        if Cd is not None:
            np.save(d / "Cd.npy", Cd[:n])
        if N is not None:
            np.save(d / "N.npy", N[:n])
        if tri is not None:
            # submesh whose faces use only the first n points; the re-index is
            # then the identity (kept points ARE the first n)
            sub = tri[(tri < n).all(axis=1)]
            np.save(d / "ptnum.npy", np.ascontiguousarray(sub.ravel()))
            out[s]["sub_faces"] = len(sub)
        out[s].update(n_points=n)
    return out


# ---- decision rule (§5.0) ---------------------------------------------------

def decide(attr: str, res: dict, present: list):
    """Apply the §5.0 rule to the (option, chunk) cells of one attribute.
    res: {(oname, chunk): {source: cell}}; present: the sources that HAVE
    this attribute (P/Cd: all five; N: 7scenes; ptnum: scannetpp)."""
    rows = []
    for oname, *_ in OPTIONS:
        for chunk in CHUNKS:
            cells = {s: c for s, c in res[(oname, chunk)].items() if "error" not in c}
            lossless = len(cells) == len(present)
            total = sum(c["stored"] for c in cells.values()) if lossless else None
            rtotal = sum(c["r_s"] for c in cells.values()) if lossless else None
            rows.append({"opt": oname, "chunk": chunk, "lossless": lossless,
                         "total": total, "rtotal": rtotal})
    valid = [r for r in rows if r["lossless"]]
    min_total = min(r["total"] for r in valid)
    in5 = [r for r in valid if r["total"] <= min_total * 1.05]
    c4 = {chunk: res[("none", chunk)] for chunk in CHUNKS}
    fast = []
    for r in in5:
        cells = {s: c for s, c in res[(r["opt"], r["chunk"])].items()
                 if "error" not in c}
        if all(c["r_s"] <= 2.0 * c4[r["chunk"]][s]["r_s"] for s, c in cells.items()):
            fast.append(r)
    c1s = [r for r in fast if r["opt"] == C1_NAME[attr]]
    pool = fast or in5          # fallback: no cell clears the 2x C4 read bar
    if c1s:
        row = min(c1s, key=lambda r: (r["total"], r["rtotal"]))
        why = "C1 survives (the convention shared with the camera filters)"
    else:
        row = min(pool, key=lambda r: (r["rtotal"], r["total"]))
        why = ("fastest read among the survivors" if fast
               else "FALLBACK: no cell clears the 2x C4 read bar; fastest in the 5 % band")
    return (row["opt"], row["chunk"]), why, rows, valid, in5, fast


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def cell_str(c) -> str:
    if "error" in c:
        return f"— ({c['error'][:40]})"
    return (f"{c['stored'] / MB:.2f} ({c['ratio']:.2f}) "
            f"{c['r_s'] * 1000:.0f} ms / w {c['w_s'] * 1000:.0f} ms")


# ---- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ply_dir", nargs="?",
                    default=str(ROOT / "_tobedelete/requirements_gt_geometry/sample_plys"))
    ap.add_argument("--write-probes", action="store_true",
                    help="write the 50k-point probes under tests/probes/geo/ (exit)")
    args = ap.parse_args()

    srcs = {}
    for p in sorted(Path(args.ply_dir).glob("*.ply")):
        s = source_of(p.name)
        P, Cd, N, tri = read_ply(p)
        srcs[s] = {"ply": p, "P": P, "Cd": Cd, "N": N, "tri": tri}
        print(f"{s}: {p.name}  M={len(P):,}  F={0 if tri is None else len(tri):,}  "
              f"Cd={'y' if Cd is not None else 'n'}  N={'y' if N is not None else 'n'}",
              flush=True)
    assert len(srcs) == 5, f"expected the 5 sample plys, got {sorted(srcs)}"

    probe_meta = {}
    if args.write_probes:
        probe_meta = write_probes(srcs)
        print(f"probes written to {PROBE_DIR}/")
        if probe_meta.get("scannetpp", {}).get("sub_faces") is not None:
            print(f"scannetpp submesh: {probe_meta['scannetpp']['sub_faces']:,} faces")
        return

    # ---- sweep ---------------------------------------------------------------
    # res[(attr, layout)][(oname, chunk)][source] = cell
    res = {("P", "flat"): {(o, c): {} for o, *_ in OPTIONS for c in CHUNKS},
           ("Cd", "flat"): {(o, c): {} for o, *_ in OPTIONS for c in CHUNKS},
           ("N", "flat"): {(o, c): {} for o, *_ in OPTIONS for c in CHUNKS},
           ("ptnum", "flat"): {(o, c): {} for o, *_ in OPTIONS for c in CHUNKS},
           ("ptnum", "F3"): {(o, c): {} for o, *_ in OPTIONS for c in CHUNKS}}
    n_cells = 0
    with tempfile.TemporaryDirectory() as td:
        fpath = Path(td) / "cell.h5"
        for s, a in srcs.items():
            jobs = [("P", "flat", a["P"])]
            if a["Cd"] is not None:
                jobs.append(("Cd", "flat", a["Cd"]))
            if a["N"] is not None:
                jobs.append(("N", "flat", a["N"]))
            if a["tri"] is not None:
                jobs.append(("ptnum", "flat", np.ascontiguousarray(a["tri"].ravel())))
                jobs.append(("ptnum", "F3", a["tri"]))
            for attr, layout, arr in jobs:
                for oname, comp, lvl, shfl in OPTIONS:
                    for chunk in CHUNKS:
                        res[(attr, layout)][(oname, chunk)][s] = bench(
                            fpath, arr, comp, lvl, shfl, chunk)
                        n_cells += 1
                        if n_cells % 36 == 0:
                            print(f"  {n_cells}/468 cells  (last: {s} {attr} {oname} "
                                  f"chunk {chunk})", flush=True)

    bad = [(k, opt, s) for k in res for opt in res[k]
           for s, c in res[k][opt].items() if "error" in c]
    if bad:
        for k, opt, s in bad[:10]:
            print(f"  ERROR cell: {k[0]} {opt} {s}: {res[k][opt][s]['error']}")

    attrs = ["P", "Cd", "N", "ptnum"]
    sources = sorted(srcs)
    present = {"P": list(sources), "Cd": list(sources),
               "N": [s for s in sources if srcs[s]["N"] is not None],
               "ptnum": [s for s in sources if srcs[s]["tri"] is not None]}
    chosen, whys, rows = {}, {}, {}
    for attr in attrs:
        ch, why, rows[attr], *_ = decide(attr, res[(attr, "flat")], present[attr])
        chosen[attr] = ch
        whys[attr] = why
        oname, chunk = ch
        tot = next(r for r in rows[attr] if (r["opt"], r["chunk"]) == ch)["total"]
        print(f"  {attr}: chosen {oname} x {chunk}  — {why}  "
              f"(total {tot / MB:.2f} MB)", flush=True)

    # ---- ptnum layout (C6) ----------------------------------------------------
    for opt_name in ("chosen", "none"):
        if opt_name == "chosen":
            key = chosen["ptnum"]
        else:
            key = ("none", 65_536)
        flat = sum(c["stored"] for c in res[("ptnum", "flat")][key].values()
                   if "error" not in c)
        f3 = sum(c["stored"] for c in res[("ptnum", "F3")][key].values()
                 if "error" not in c)
        print(f"  ptnum layout at {key}: flat {flat / MB:.2f} MB vs (F,3) {f3 / MB:.2f} MB")

    # ---- h5/ply ratios and the 43-scene projection ------------------------------
    ratios = {}
    for s in sources:
        h5 = 0
        for attr in ("P", "Cd", "N", "ptnum"):
            c = res[(attr, "flat")][chosen[attr]].get(s)
            if c and "error" not in c:
                h5 += c["stored"]
        ply = srcs[s]["ply"].stat().st_size
        ratios[s] = {"h5": h5, "ply": ply, "ratio": h5 / ply,
                     "total": SRC_TOTAL_BYTES[s],
                     "projected": h5 / ply * SRC_TOTAL_BYTES[s]}
    proj_total = sum(r["projected"] for r in ratios.values())
    print(f"  projected 43-scene total: {proj_total / MB:.0f} MB "
          f"({'OK < 1.8 GB' if proj_total < 1.8e9 else 'FAIL >= 1.8 GB'})")

    # ---- C7 per-source exception (plain gzip for 7scenes P) ----------------------
    chunk = chosen["P"][1]
    c7 = {}
    for oname in ("gzip4", "gzip9"):
        c = res[("P", "flat")][(oname, chunk)]["7scenes"]
        c7[oname] = c["stored"] if "error" not in c else None
    ch_p = res[("P", "flat")][chosen["P"]]["7scenes"]["stored"]
    best_plain = min(v for v in c7.values() if v)
    saving = (ch_p - best_plain) / ratios["7scenes"]["ply"] * ratios["7scenes"]["total"]
    c7["saving"] = saving
    c7["threshold"] = 0.10 * proj_total
    print(f"  C7: plain gzip saves {saving / MB:.1f} MB projected "
          f"(threshold {c7['threshold'] / MB:.1f} MB) -> "
          f"{'adopt' if saving > c7['threshold'] else 'NOT worth a per-source policy'}")

    # ---- C8 info-only ------------------------------------------------------------
    c8 = {}
    import importlib.util
    c8["zstd"] = "hdf5plugin installed" if importlib.util.find_spec("hdf5plugin") \
        else "hdf5plugin NOT installed (zstd not measured; new dependency, ineligible)"
    oname, chunk = chosen["N"]
    idx = [o[0] for o in OPTIONS].index(oname)
    N16 = srcs["7scenes"]["N"].astype(np.float16)
    with tempfile.TemporaryDirectory() as td:
        fpath = Path(td) / "info.h5"
        c = bench(fpath, N16, OPTIONS[idx][1], OPTIONS[idx][2], OPTIONS[idx][3], chunk)
        c8["N_f16"] = {"cell": c,
                       "max_err": float(np.abs(N16.astype(np.float32) - srcs["7scenes"]["N"]).max())}
    c8["P_quant"] = {}
    oname, chunk = chosen["P"]
    o = OPTIONS[[o[0] for o in OPTIONS].index(oname)]
    with tempfile.TemporaryDirectory() as td:
        fpath = Path(td) / "info.h5"
        for s in sources:
            Pq = (np.round(srcs[s]["P"] * 1000.0) / 1000.0).astype(np.float32)
            c = bench(fpath, Pq, o[1], o[2], o[3], chunk)
            c8["P_quant"][s] = {"cell": c,
                                "max_err": float(np.abs(Pq - srcs[s]["P"]).max())}

    render(srcs, res, chosen, whys, rows, ratios, proj_total, c7, c8, present)
    print(f"report written to {REPORT}")


# ---- report ------------------------------------------------------------------

def render(srcs, res, chosen, whys, rows, ratios, proj_total, c7, c8, present):
    L = []
    a = L.append
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    a("# geo compression probes & sweep (H5Geo)")
    a("")
    a(f"Compression experiment from `requirements_gt_geometry/IMPLEMENTATION_SPEC.md` "
      f"§5, run {now} on the 5 sample plys in "
      f"`requirements_gt_geometry/sample_plys/`. One-off benchmark, not in the pytest "
      f"suite. Rerun: `pixi run smoke-geo [ply_dir]` (add `--write-probes` to "
      f"regenerate the probe npys).")
    a("")
    a("**Chosen policy (decision rule §5.0):**")
    a("")
    for attr in ("P", "Cd", "N", "ptnum"):
        oname, chunk = chosen[attr]
        a(f"- `{attr}`: **{oname}**, chunk rows **{chunk}** — {whys[attr]}.")
    a(f"- ptnum layout: **flat (V,)** (C6, below).")
    a(f"- Projected 43-scene total: **{proj_total / MB:.0f} MB** "
      f"({'< 1.8 GB, pass' if proj_total < 1.8e9 else '>= 1.8 GB, FAIL'}).")
    a("")

    attrs = [("P", "all sources"), ("Cd", "all sources"),
             ("N", "7scenes only (the only GT with normals)"),
             ("ptnum", "scannetpp only (the only GT with faces)")]
    for attr, note in attrs:
        src_present = present[attr]
        a(f"## {attr}  ({note})")
        a("")
        for chunk in CHUNKS:
            a(f"### sweep, chunk rows {chunk}")
            a("")
            a(f"cells: stored MB (ratio to raw) · read ms / write ms; raw is the "
              f"f32/u8/i32 array the H5Geo writers will store.")
            a("")
            hdr = ["option"] + [s for s in src_present]
            body = []
            for oname, *_ in OPTIONS:
                row = [("*" if oname == C1_NAME[attr] else "") + oname]
                row += [cell_str(res[(attr, "flat")][(oname, chunk)][s])
                        for s in src_present]
                body.append(row)
            a(md_table(hdr, body))
            a("")
        a(f"### policy — decision rule §5.0 (totals over the {len(src_present)} "
          f"sources above)")
        a("")
        a("`*` = C1; min total = "
          f"{min(r['total'] for r in rows[attr] if r['lossless']) / MB:.2f} MB.")
        a("")
        hdr = ["option", "chunk", "total MB", "total read ms", "lossless",
               "≤ 5 %", "read ≤ 2× C4", "chosen"]
        valid = [r for r in rows[attr] if r["lossless"]]
        min_total = min(r["total"] for r in valid)
        in5 = [r for r in valid if r["total"] <= min_total * 1.05]
        fast = [r for r in in5
                if all(c["r_s"] <= 2.0 * res[(attr, "flat")][("none", r["chunk"])][s]["r_s"]
                       for s, c in res[(attr, "flat")][r["opt"], r["chunk"]].items()
                       if "error" not in c)]
        ch = chosen[attr]
        body = []
        for r in rows[attr]:
            mark = "**chosen**" if (r["opt"], r["chunk"]) == ch else ""
            body.append([("*" if r["opt"] == C1_NAME[attr] else "") + r["opt"],
                         str(r["chunk"]),
                         f"{r['total'] / MB:.2f}" if r["total"] else "—",
                         f"{r['rtotal'] * 1000:.0f}" if r["rtotal"] else "—",
                         "y" if r["lossless"] else "n",
                         "y" if r in in5 else "",
                         "y" if r in fast else "",
                         mark])
        a(md_table(hdr, body))
        a("")

    a("## ptnum layout (C6): flat (V,) vs (F,3)")
    a("")
    a("Same byte content, two layouts. Stored MB at the chosen filter and at raw:")
    a("")
    hdr = ["filter / chunk", "flat (V,)", "(F,3)"]
    body = []
    for key in (chosen["ptnum"], ("none", 65_536)):
        flat = sum(c["stored"] for c in res[("ptnum", "flat")][key].values()
                   if "error" not in c)
        f3 = sum(c["stored"] for c in res[("ptnum", "F3")][key].values()
                 if "error" not in c)
        label = f"{key[0]} × {key[1]}"
        body.append([label, f"{flat / MB:.2f}", f"{f3 / MB:.2f}"])
    a(md_table(hdr, body))
    a("")
    flat_tot = sum(c["stored"] for c in res[("ptnum", "flat")][chosen["ptnum"]].values()
                   if "error" not in c)
    f3_tot = sum(c["stored"] for c in res[("ptnum", "F3")][chosen["ptnum"]].values()
                 if "error" not in c)
    a(f"Verdict: flat {flat_tot / MB:.2f} MB vs (F,3) {f3_tot / MB:.2f} MB — "
      f"{'equal (diff < 0.5 %), keep the spec layout flat (V,)' if abs(flat_tot - f3_tot) / flat_tot < 0.005 else 'they differ; adopt the smaller'}.")
    a("")

    a("## per-source h5/ply ratio and 43-scene projection")
    a("")
    a("Per source: h5 bytes = sum of the chosen-policy cells for the attributes that "
      "source has (P always; Cd; N for 7scenes; ptnum for scannetpp). Ply bytes = "
      "the sample file on disk; the 43-scene total per source is from "
      "`REQUIREMENTS_GT_GEOMETRY.md` (~1.8 GB; scannetpp = the remainder, estimate).")
    a("")
    hdr = ["source", "sample ply MB", "h5 MB", "ratio", "43-scene total MB",
           "projected MB"]
    body = [[s, f"{ratios[s]['ply'] / MB:.1f}", f"{ratios[s]['h5'] / MB:.2f}",
             f"{ratios[s]['ratio']:.2f}", f"{ratios[s]['total'] / MB:.0f}",
             f"{ratios[s]['projected'] / MB:.0f}"] for s in sorted(ratios)]
    body.append(["**total**", "", "", "",
                 f"{sum(r['total'] for r in ratios.values()) / MB:.0f}",
                 f"**{proj_total / MB:.0f}**"])
    a(md_table(hdr, body))
    a("")
    a(f"Global check (§5.0 rule 6): {proj_total / MB:.0f} MB "
      f"{'< 1.8 GB, pass' if proj_total < 1.8e9 else '>= 1.8 GB, FAIL'} "
      f"(expected ≈ 0.8 GB, measured {proj_total / 1e9:.2f} GB).")
    a("")

    a("## C7 per-source exception (plain gzip for grid-like P)")
    a("")
    a(f"7scenes P lies on an f32-exact grid, where shuffle (which breaks long "
      f"identical runs) hurts. At the chosen chunk rows {chosen['P'][1]}: "
      f"chosen-policy cell {res[('P', 'flat')][chosen['P']]['7scenes']['stored'] / MB:.2f} MB, "
      f"plain gzip4 {c7['gzip4'] / MB:.2f} MB, plain gzip9 {c7['gzip9'] / MB:.2f} MB. "
      f"Projected saving **{c7['saving'] / MB:.1f} MB** vs the 10 %-of-projected-total "
      f"threshold **{c7['threshold'] / MB:.1f} MB** → "
      f"{'adopt a per-source exception' if c7['saving'] > c7['threshold'] else '**not worth it**, keep one global policy'}.")
    a("")

    a("## C8 info-only (ineligible: lossy or a new dependency)")
    a("")
    hdr = ["option", "stored MB", "max abs err", "note"]
    n16 = c8["N_f16"]
    body = [["N as f16 (7scenes)",
             f"{n16['cell']['stored'] / MB:.2f}" if "error" not in n16["cell"] else "—",
             f"{n16['max_err']:.1e}", "lossy, ineligible"]]
    for s in sorted(c8["P_quant"]):
        q = c8["P_quant"][s]
        body.append([f"P quantised to 1e-3 m grid ({s})",
                     f"{q['cell']['stored'] / MB:.2f}" if "error" not in q["cell"] else "—",
                     f"{q['max_err']:.1e}", "lossy, ineligible"])
    body.append(["zstd via hdf5plugin", "—", "—", c8["zstd"]])
    a(md_table(hdr, body))
    a("")

    a("## probes (50k points, §5.3)")
    a("")
    a("Written once by `pixi run smoke-geo --write-probes`; no ply parser needed at "
      "test time. scannetpp: the first 50k points plus the submesh whose faces use "
      "only those points (re-indexed to identity).")
    a("")
    hdr = ["source", "P.npy", "Cd.npy", "N.npy", "ptnum.npy"]
    body = []
    for s in sorted(srcs):
        cells = []
        for fn in ("P.npy", "Cd.npy", "N.npy", "ptnum.npy"):
            p = PROBE_DIR / s / fn
            cells.append(f"{np.load(p, mmap_mode='r').shape} {np.load(p, mmap_mode='r').dtype}"
                         if p.exists() else "—")
        body.append([s] + cells)
    a(md_table(hdr, body))
    a("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
