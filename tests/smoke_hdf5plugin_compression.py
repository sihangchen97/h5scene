"""Zstd / Blosc / LZ4 (hdf5plugin) compression test for the H5Geo + H5Cam
filter choice — the "C8 info only" options from
requirements_gt_geometry/IMPLEMENTATION_SPEC.md §5.0, now measured on request
(2026-11-13) instead of skipped as a new dependency.

Same protocol as smoke_geo_compression.py (stored bytes, ratio, write time,
full-read time ds[:] = the consumer's only access pattern, bit-exact round
trip), over:
  - the 50k-point geo probes: tests/probes/geo/<source>/{P,Cd,N,ptnum}.npy
    (P/Cd f32/u8 (50k,3); N 7scenes only; ptnum scannetpp only, 1-D i32)
  - the depth probe: tests/probes/adt_work_skeleton_seq137/frame6_depth_mm.npy
    (uint16 640x640, 0 = invalid — the real H5Cam depth layout)

hdf5plugin 7.x API: hdf5plugin.Zstd / Blosc2 / Blosc / LZ4 (lazy attributes,
no submodules). Requires the pixi `plugin` env:  pixi run smoke-plugin
Report: tests/probes/hdf5plugin/README.md
"""
from __future__ import annotations

import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from hdf5plugin import Blosc, Blosc2, LZ4, Zstd

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "tests" / "probes" / "geo"
DEPTH = ROOT / "tests" / "probes" / "adt_work_skeleton_seq137" / "frame6_depth_mm.npy"
REPORT = ROOT / "tests" / "probes" / "hdf5plugin" / "README.md"
MB = 1e6

# (label, h5py create_dataset filter kwargs)
FILTERS = [
    ("none", {}),
    ("gzip4+sh (h5py baseline)", dict(compression="gzip", compression_opts=4,
                                      shuffle=True)),
    ("zstd1", dict(compression=Zstd(1))),
    ("zstd3", dict(compression=Zstd(3))),
    ("zstd9", dict(compression=Zstd(9))),
    ("blosc2-lz4 c5", dict(compression=Blosc2(cname="lz4", clevel=5, filters=0))),
    ("blosc2-lz4 c5 shuf", dict(compression=Blosc2(cname="lz4", clevel=5,
                                                    filters=Blosc2.SHUFFLE))),
    ("blosc2-zstd c3", dict(compression=Blosc2(cname="zstd", clevel=3, filters=0))),
    ("blosc2-zlib c5", dict(compression=Blosc2(cname="zlib", clevel=5, filters=0))),
    ("blosc1-lz4 c5 shuf", dict(compression=Blosc(cname="lz4", clevel=5, shuffle=1))),
    ("blosc1-lz4 c5 noshuf", dict(compression=Blosc(cname="lz4", clevel=5, shuffle=0))),
    ("lz4 (hdf5plugin)", dict(compression=LZ4())),
]

# (label, path, chunk) — chunk rows 16384 like the geo sweep; depth = full frame
def datasets() -> list[tuple[str, Path, tuple]]:
    out = []
    for s in sorted(d.name for d in GEO.iterdir() if d.is_dir()):
        for fn, cols in (("P.npy", 3), ("Cd.npy", 3), ("N.npy", 3), ("ptnum.npy", 1)):
            p = GEO / s / fn
            if p.exists():
                ch = (16_384, cols) if cols > 1 else (16_384,)
                out.append((f"geo/{s} {fn[:-4]}", p, ch))
    out.append(("depth frame6 (uint16 640x640)", DEPTH, (640, 640)))
    return out


def bench(arr: np.ndarray, kw: dict, chunk: tuple) -> dict:
    t0 = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.h5"
            f = h5py.File(p, "w")
            d = f.create_dataset("x", data=arr, chunks=chunk, **kw)
            f.flush()
            stored = d.id.get_storage_size()
            f.close()
            with h5py.File(p, "r") as f:
                d = f["x"]
                d[:]                                          # warm
                r_s, ok = float("inf"), False
                for _ in range(3):
                    t0 = time.perf_counter()
                    ok = np.array_equal(d[:], arr)
                    r_s = min(r_s, time.perf_counter() - t0)
        if not ok:
            return {"error": "ROUND-TRIP MISMATCH"}
        return {"stored": stored, "r_s": r_s}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> None:
    raw = {d[0]: np.load(d[1]) for d in datasets()}
    print(f"{'dataset':<34s} {'raw MB':>7s}  filter cells...", flush=True)
    res = {}
    for label, p, chunk in datasets():
        arr = raw[label]      # kept in `raw` for the report tables below
        for fl, kw in FILTERS:
            res[(label, fl)] = bench(arr, kw, chunk)
            c = res[(label, fl)]
            print(f"  {label:<32s} {fl:<22s} "
                  f"{c.get('stored', 0) / MB:7.3f} MB  "
                  f"r {c.get('r_s', 0) * 1000:7.2f} ms"
                  if "error" not in c else f"  {label} {fl}: {c['error']}")
            del c
        del arr

    L = []
    a = L.append
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    a("# hdf5plugin (zstd / blosc / lz4) compression test")
    a("")
    a(f"Run {now} via `pixi run smoke-plugin` (pixi `plugin` env, "
      f"hdf5plugin 7.x, h5py {h5py.__version__}). Same protocol as "
      f"`tests/probes/geo/README.md`: stored MB, ratio to raw, write time "
      f"omitted here (not decisive), full-read time = `ds[:]` (the consumer's "
      f"only access pattern), bit-exact round trip. Chunks: 16384 rows; "
      f"depth = full 640x640 frame.")
    a("")
    for label, p, chunk in datasets():
        arr = raw[label]
        a(f"## {label}  (raw {arr.nbytes / MB:.2f} MB, dtype {arr.dtype})")
        a("")
        base = res[(label, "gzip4+sh (h5py baseline)")]
        hdr = ["filter", "stored MB", "ratio", "read ms", "vs gzip4+sh (stored)", "vs raw (read)", "round trip"]
        body = []
        for fl, _ in FILTERS:
            c = res[(label, fl)]
            if "error" in c:
                body.append([fl, "—", "—", "—", "—", "—", f"**{c['error'][:60]}**"])
                continue
            vs = c["stored"] / base["stored"]
            body.append([fl, f"{c['stored'] / MB:.3f}", f"{c['stored'] / arr.nbytes:.3f}",
                         f"{c['r_s'] * 1000:.2f}", f"{vs:.2f}x",
                         f"{c['r_s'] / res[(label, 'none')]['r_s']:.2f}x", "ok"])
        a("| " + " | ".join(hdr) + " |")
        a("|" + "---|" * len(hdr))
        a("\n".join("| " + " | ".join(r) + " |" for r in body))
        a("")

    a("## reading the numbers")
    a("")
    a("Per-attribute one-line verdict (written after the table):")
    a("")
    for label, *_ in datasets():
        z = min((res[(label, f)] for f, _ in FILTERS if f.startswith(("zstd", "blosc", "lz4"))
                and "error" not in res[(label, f)]),
                key=lambda c: c["stored"], default=None)
        g = res[(label, "gzip4+sh (h5py baseline)")]
        if z is None:
            a(f"- {label}: no plugin filter worked.")
            continue
        a(f"- {label}: smallest plugin cell {z['stored'] / MB:.3f} MB "
          f"({z['stored'] / g['stored']:.2f}x the gzip4+sh cell "
          f"{g['stored'] / MB:.3f} MB), read {z['r_s'] * 1000:.2f} ms vs "
          f"gzip4+sh {g['r_s'] * 1000:.2f} ms.")
    a("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    sys.exit(main())
