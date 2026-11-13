# geo compression probes & sweep (H5Geo)

Compression experiment from `requirements_gt_geometry/IMPLEMENTATION_SPEC.md` §5, run 2026-11-13 15:04 NZDT on the 5 sample plys in `requirements_gt_geometry/sample_plys/`. One-off benchmark, not in the pytest suite. Rerun: `pixi run smoke-geo [ply_dir]` (add `--write-probes` to regenerate the probe npys).

**Chosen policy (decision rule §5.0):**

- `P`: **gzip4+sh** (C1), chunk rows **262144** — owner decision: keep C1 for consistency with `/depth` and the camera convention; the sweep's fallback pick (gzip6+sh) was only marginally faster (103 vs 113 ms total read) and 0.3 MB smaller — not worth breaking the convention.
- `Cd`: **gzip4**, chunk rows **262144** — C1 survives (the convention shared with the camera filters).
- `N`: **gzip4+sh** (C1), chunk rows **262144** — owner decision as for P (sweep fallback was gzip6+sh).
- `ptnum`: **gzip4+sh** (C1), chunk rows **65536** — owner decision as for P (sweep fallback was gzip1+sh).
- ptnum layout: **flat (V,)** (C6, below).
- Projected 43-scene total: **811 MB** (< 1.8 GB, pass).

## P  (all sources)

### sweep, chunk rows 16384

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | 7scenes | dtu | hiroom | nrgbd | scannetpp |
|---|---|---|---|---|---|
| none | 12.19 (1.02) 2 ms / w 3 ms | 12.19 (1.02) 6 ms / w 6 ms | 14.55 (1.01) 2 ms / w 3 ms | 12.19 (1.02) 2 ms / w 3 ms | 4.33 (1.03) 0 ms / w 1 ms |
| none+sh | 12.19 (1.02) 5 ms / w 6 ms | 12.19 (1.02) 6 ms / w 9 ms | 14.55 (1.01) 7 ms / w 8 ms | 12.19 (1.02) 7 ms / w 8 ms | 4.33 (1.03) 0 ms / w 3 ms |
| gzip1 | 7.35 (0.61) 50 ms / w 207 ms | 11.12 (0.93) 44 ms / w 335 ms | 12.86 (0.89) 57 ms / w 359 ms | 11.07 (0.92) 48 ms / w 391 ms | 3.35 (0.80) 0 ms / w 99 ms |
| gzip1+sh | 8.56 (0.71) 60 ms / w 241 ms | 10.08 (0.84) 27 ms / w 203 ms | 11.52 (0.80) 43 ms / w 258 ms | 10.16 (0.85) 28 ms / w 205 ms | 2.86 (0.68) 0 ms / w 79 ms |
| gzip4 | 6.81 (0.57) 46 ms / w 251 ms | 11.09 (0.92) 44 ms / w 371 ms | 12.90 (0.89) 57 ms / w 452 ms | 11.02 (0.92) 42 ms / w 368 ms | 3.33 (0.79) 0 ms / w 132 ms |
| *gzip4+sh | 8.38 (0.70) 56 ms / w 279 ms | 9.92 (0.83) 27 ms / w 258 ms | 11.30 (0.78) 39 ms / w 300 ms | 10.05 (0.84) 23 ms / w 230 ms | 2.82 (0.67) 0 ms / w 96 ms |
| gzip6 | 6.76 (0.56) 39 ms / w 493 ms | 11.09 (0.92) 44 ms / w 367 ms | 12.90 (0.89) 56 ms / w 494 ms | 11.02 (0.92) 39 ms / w 332 ms | 3.32 (0.79) 0 ms / w 193 ms |
| gzip6+sh | 8.31 (0.69) 51 ms / w 482 ms | 9.83 (0.82) 26 ms / w 546 ms | 11.19 (0.78) 34 ms / w 469 ms | 9.97 (0.83) 22 ms / w 587 ms | 2.81 (0.67) 0 ms / w 134 ms |
| gzip9 | 6.76 (0.56) 39 ms / w 484 ms | 11.09 (0.92) 40 ms / w 341 ms | 12.90 (0.89) 50 ms / w 453 ms | 11.02 (0.92) 39 ms / w 331 ms | 3.32 (0.79) 0 ms / w 231 ms |
| gzip9+sh | 8.29 (0.69) 50 ms / w 892 ms | 9.80 (0.82) 24 ms / w 1936 ms | 11.16 (0.77) 33 ms / w 1493 ms | 9.95 (0.83) 22 ms / w 2379 ms | 2.81 (0.67) 0 ms / w 229 ms |
| lzf | 9.40 (0.78) 28 ms / w 55 ms | 12.00 (1.00) 1 ms / w 47 ms | 14.40 (1.00) 7 ms / w 61 ms | 12.00 (1.00) 3 ms / w 59 ms | 4.05 (0.97) 0 ms / w 26 ms |
| lzf+sh | 10.18 (0.85) 27 ms / w 68 ms | 11.01 (0.92) 17 ms / w 50 ms | 12.62 (0.87) 24 ms / w 62 ms | 11.07 (0.92) 21 ms / w 63 ms | 3.21 (0.76) 0 ms / w 18 ms |

### sweep, chunk rows 65536

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | 7scenes | dtu | hiroom | nrgbd | scannetpp |
|---|---|---|---|---|---|
| none | 12.58 (1.05) 2 ms / w 2 ms | 12.58 (1.05) 5 ms / w 6 ms | 14.94 (1.04) 3 ms / w 4 ms | 12.58 (1.05) 2 ms / w 3 ms | 4.72 (1.12) 0 ms / w 1 ms |
| none+sh | 12.58 (1.05) 5 ms / w 6 ms | 12.58 (1.05) 7 ms / w 7 ms | 14.94 (1.04) 7 ms / w 7 ms | 12.58 (1.05) 7 ms / w 8 ms | 4.72 (1.12) 0 ms / w 3 ms |
| gzip1 | 7.33 (0.61) 49 ms / w 197 ms | 11.12 (0.93) 42 ms / w 309 ms | 12.85 (0.89) 56 ms / w 343 ms | 11.07 (0.92) 47 ms / w 336 ms | 3.35 (0.80) 0 ms / w 100 ms |
| gzip1+sh | 8.46 (0.71) 58 ms / w 249 ms | 10.06 (0.84) 28 ms / w 218 ms | 11.45 (0.79) 41 ms / w 274 ms | 10.16 (0.85) 26 ms / w 218 ms | 2.85 (0.68) 0 ms / w 79 ms |
| gzip4 | 6.77 (0.56) 44 ms / w 254 ms | 11.09 (0.92) 44 ms / w 386 ms | 12.90 (0.89) 56 ms / w 463 ms | 11.02 (0.92) 42 ms / w 383 ms | 3.33 (0.79) 0 ms / w 134 ms |
| *gzip4+sh | 8.27 (0.69) 53 ms / w 288 ms | 9.92 (0.83) 26 ms / w 263 ms | 11.25 (0.78) 39 ms / w 315 ms | 10.05 (0.84) 22 ms / w 231 ms | 2.81 (0.67) 0 ms / w 98 ms |
| gzip6 | 6.73 (0.56) 36 ms / w 495 ms | 11.09 (0.92) 43 ms / w 376 ms | 12.90 (0.89) 56 ms / w 519 ms | 11.02 (0.92) 38 ms / w 344 ms | 3.32 (0.79) 0 ms / w 202 ms |
| gzip6+sh | 8.21 (0.68) 49 ms / w 399 ms | 9.82 (0.82) 24 ms / w 577 ms | 11.15 (0.77) 33 ms / w 465 ms | 9.97 (0.83) 21 ms / w 618 ms | 2.81 (0.67) 0 ms / w 149 ms |
| gzip9 | 6.73 (0.56) 37 ms / w 505 ms | 11.09 (0.92) 39 ms / w 345 ms | 12.90 (0.89) 49 ms / w 480 ms | 11.02 (0.92) 38 ms / w 342 ms | 3.32 (0.79) 0 ms / w 241 ms |
| gzip9+sh | 8.18 (0.68) 48 ms / w 1125 ms | 9.77 (0.81) 21 ms / w 2556 ms | 11.10 (0.77) 32 ms / w 1870 ms | 9.92 (0.83) 26 ms / w 3113 ms | 2.80 (0.67) 0 ms / w 259 ms |
| lzf | 9.40 (0.78) 28 ms / w 60 ms | 12.01 (1.00) 1 ms / w 45 ms | 14.42 (1.00) 6 ms / w 59 ms | 12.01 (1.00) 2 ms / w 58 ms | 4.06 (0.97) 0 ms / w 20 ms |
| lzf+sh | 10.17 (0.85) 28 ms / w 72 ms | 11.01 (0.92) 17 ms / w 47 ms | 12.62 (0.87) 25 ms / w 60 ms | 11.06 (0.92) 21 ms / w 58 ms | 3.21 (0.76) 0 ms / w 18 ms |

### sweep, chunk rows 262144

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | 7scenes | dtu | hiroom | nrgbd | scannetpp |
|---|---|---|---|---|---|
| none | 12.58 (1.05) 1 ms / w 2 ms | 12.58 (1.05) 4 ms / w 5 ms | 15.73 (1.09) 2 ms / w 3 ms | 12.58 (1.05) 2 ms / w 3 ms | 6.29 (1.50) 0 ms / w 1 ms |
| none+sh | 12.58 (1.05) 4 ms / w 6 ms | 12.58 (1.05) 6 ms / w 9 ms | 15.73 (1.09) 6 ms / w 9 ms | 12.58 (1.05) 5 ms / w 7 ms | 6.29 (1.50) 0 ms / w 3 ms |
| gzip1 | 7.32 (0.61) 39 ms / w 198 ms | 11.12 (0.93) 36 ms / w 327 ms | 12.85 (0.89) 52 ms / w 359 ms | 11.07 (0.92) 37 ms / w 339 ms | 3.36 (0.80) 0 ms / w 101 ms |
| gzip1+sh | 8.43 (0.70) 47 ms / w 243 ms | 10.04 (0.84) 22 ms / w 222 ms | 11.44 (0.79) 37 ms / w 270 ms | 10.13 (0.84) 21 ms / w 214 ms | 2.85 (0.68) 0 ms / w 79 ms |
| gzip4 | 6.77 (0.56) 35 ms / w 256 ms | 11.09 (0.92) 35 ms / w 386 ms | 12.90 (0.89) 49 ms / w 473 ms | 11.02 (0.92) 31 ms / w 364 ms | 3.33 (0.79) 0 ms / w 138 ms |
| *gzip4+sh | 8.24 (0.69) 41 ms / w 279 ms | 9.92 (0.83) 21 ms / w 261 ms | 11.23 (0.78) 34 ms / w 317 ms | 10.05 (0.84) 18 ms / w 233 ms | 2.81 (0.67) 0 ms / w 103 ms |
| gzip6 | 6.72 (0.56) 29 ms / w 494 ms | 11.09 (0.92) 34 ms / w 379 ms | 12.90 (0.89) 48 ms / w 521 ms | 11.02 (0.92) 31 ms / w 356 ms | 3.32 (0.79) 0 ms / w 204 ms |
| gzip6+sh | 8.18 (0.68) 39 ms / w 403 ms | 9.82 (0.82) 19 ms / w 569 ms | 11.12 (0.77) 29 ms / w 466 ms | 9.97 (0.83) 17 ms / w 626 ms | 2.80 (0.67) 0 ms / w 154 ms |
| gzip9 | 6.72 (0.56) 29 ms / w 510 ms | 11.09 (0.92) 30 ms / w 334 ms | 12.90 (0.89) 43 ms / w 476 ms | 11.02 (0.92) 31 ms / w 346 ms | 3.32 (0.79) 0 ms / w 239 ms |
| gzip9+sh | 8.15 (0.68) 39 ms / w 1184 ms | 9.76 (0.81) 17 ms / w 2714 ms | 11.08 (0.77) 28 ms / w 1968 ms | 9.91 (0.83) 21 ms / w 3623 ms | 2.80 (0.67) 2 ms / w 283 ms |
| lzf | 9.40 (0.78) 26 ms / w 60 ms | 12.07 (1.01) 1 ms / w 45 ms | 14.43 (1.00) 5 ms / w 59 ms | 12.07 (1.01) 1 ms / w 56 ms | 4.07 (0.97) 0 ms / w 21 ms |
| lzf+sh | 10.16 (0.85) 21 ms / w 76 ms | 11.00 (0.92) 12 ms / w 47 ms | 12.62 (0.87) 22 ms / w 60 ms | 11.06 (0.92) 16 ms / w 58 ms | 3.22 (0.77) 0 ms / w 19 ms |

### policy — decision rule §5.0 (totals over the 5 sources above)

`*` = C1; min total = 41.69 MB.

| option | chunk | total MB | total read ms | lossless | ≤ 5 % | read ≤ 2× C4 | chosen |
|---|---|---|---|---|---|---|---|
| none | 16384 | 55.44 | 12 | y |  |  |  |
| none | 65536 | 57.41 | 11 | y |  |  |  |
| none | 262144 | 59.77 | 10 | y |  |  |  |
| none+sh | 16384 | 55.44 | 26 | y |  |  |  |
| none+sh | 65536 | 57.41 | 26 | y |  |  |  |
| none+sh | 262144 | 59.77 | 21 | y |  |  |  |
| gzip1 | 16384 | 45.73 | 199 | y |  |  |  |
| gzip1 | 65536 | 45.72 | 195 | y |  |  |  |
| gzip1 | 262144 | 45.72 | 164 | y |  |  |  |
| gzip1+sh | 16384 | 43.17 | 158 | y | y |  |  |
| gzip1+sh | 65536 | 42.99 | 154 | y | y |  |  |
| gzip1+sh | 262144 | 42.90 | 127 | y | y |  |  |
| gzip4 | 16384 | 45.14 | 190 | y |  |  |  |
| gzip4 | 65536 | 45.11 | 186 | y |  |  |  |
| gzip4 | 262144 | 45.10 | 151 | y |  |  |  |
| *gzip4+sh | 16384 | 42.46 | 146 | y | y |  |  |
| *gzip4+sh | 65536 | 42.31 | 140 | y | y |  |  |
| *gzip4+sh | 262144 | 42.25 | 113 | y | y |  |  |
| gzip6 | 16384 | 45.10 | 178 | y |  |  |  |
| gzip6 | 65536 | 45.06 | 173 | y |  |  |  |
| gzip6 | 262144 | 45.05 | 143 | y |  |  |  |
| gzip6+sh | 16384 | 42.11 | 133 | y | y |  |  |
| gzip6+sh | 65536 | 41.95 | 128 | y | y |  |  |
| gzip6+sh | 262144 | 41.89 | 103 | y | y |  | **chosen** |
| gzip9 | 16384 | 45.09 | 168 | y |  |  |  |
| gzip9 | 65536 | 45.06 | 163 | y |  |  |  |
| gzip9 | 262144 | 45.05 | 134 | y |  |  |  |
| gzip9+sh | 16384 | 42.00 | 129 | y | y |  |  |
| gzip9+sh | 65536 | 41.78 | 127 | y | y |  |  |
| gzip9+sh | 262144 | 41.69 | 106 | y | y |  |  |
| lzf | 16384 | 51.87 | 39 | y |  |  |  |
| lzf | 65536 | 51.90 | 38 | y |  |  |  |
| lzf | 262144 | 52.05 | 33 | y |  |  |  |
| lzf+sh | 16384 | 48.08 | 89 | y |  |  |  |
| lzf+sh | 65536 | 48.06 | 91 | y |  |  |  |
| lzf+sh | 262144 | 48.08 | 71 | y |  |  |  |

## Cd  (all sources)

### sweep, chunk rows 16384

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | 7scenes | dtu | hiroom | nrgbd | scannetpp |
|---|---|---|---|---|---|
| none | 3.05 (1.02) 0 ms / w 2 ms | 3.05 (1.02) 0 ms / w 1 ms | 3.64 (1.01) 0 ms / w 2 ms | 3.05 (1.02) 0 ms / w 1 ms | 1.08 (1.03) 0 ms / w 1 ms |
| none+sh | 3.05 (1.02) 0 ms / w 1 ms | 3.05 (1.02) 0 ms / w 1 ms | 3.64 (1.01) 0 ms / w 1 ms | 3.05 (1.02) 0 ms / w 1 ms | 1.08 (1.03) 0 ms / w 0 ms |
| gzip1 | 2.52 (0.84) 1 ms / w 63 ms | 2.79 (0.93) 0 ms / w 49 ms | 2.35 (0.65) 0 ms / w 49 ms | 2.34 (0.78) 0 ms / w 51 ms | 0.87 (0.83) 0 ms / w 17 ms |
| gzip1+sh | 2.52 (0.84) 0 ms / w 53 ms | 2.79 (0.93) 0 ms / w 50 ms | 2.35 (0.65) 0 ms / w 53 ms | 2.34 (0.78) 0 ms / w 52 ms | 0.87 (0.83) 0 ms / w 17 ms |
| *gzip4 | 2.56 (0.85) 0 ms / w 60 ms | 2.84 (0.95) 0 ms / w 59 ms | 2.36 (0.65) 1 ms / w 87 ms | 2.50 (0.83) 0 ms / w 82 ms | 0.87 (0.83) 0 ms / w 20 ms |
| gzip4+sh | 2.56 (0.85) 0 ms / w 61 ms | 2.84 (0.95) 0 ms / w 58 ms | 2.36 (0.65) 1 ms / w 86 ms | 2.50 (0.83) 0 ms / w 82 ms | 0.87 (0.83) 0 ms / w 20 ms |
| gzip6 | 2.56 (0.85) 1 ms / w 64 ms | 2.83 (0.94) 0 ms / w 66 ms | 2.33 (0.65) 0 ms / w 133 ms | 2.50 (0.83) 0 ms / w 96 ms | 0.87 (0.83) 0 ms / w 22 ms |
| gzip6+sh | 2.56 (0.85) 0 ms / w 66 ms | 2.83 (0.94) 0 ms / w 67 ms | 2.33 (0.65) 1 ms / w 124 ms | 2.50 (0.83) 0 ms / w 97 ms | 0.87 (0.83) 0 ms / w 22 ms |
| gzip9 | 2.56 (0.85) 0 ms / w 68 ms | 2.83 (0.94) 0 ms / w 100 ms | 2.32 (0.64) 0 ms / w 485 ms | 2.50 (0.83) 0 ms / w 102 ms | 0.87 (0.83) 0 ms / w 26 ms |
| gzip9+sh | 2.56 (0.85) 0 ms / w 71 ms | 2.83 (0.94) 1 ms / w 110 ms | 2.32 (0.64) 1 ms / w 551 ms | 2.50 (0.83) 0 ms / w 99 ms | 0.87 (0.83) 0 ms / w 26 ms |
| lzf | 2.87 (0.96) 0 ms / w 44 ms | 2.98 (0.99) 0 ms / w 22 ms | 2.81 (0.78) 1 ms / w 33 ms | 2.77 (0.92) 0 ms / w 22 ms | 1.00 (0.95) 0 ms / w 7 ms |
| lzf+sh | 2.87 (0.96) 0 ms / w 30 ms | 2.98 (0.99) 1 ms / w 23 ms | 2.81 (0.78) 1 ms / w 32 ms | 2.77 (0.92) 0 ms / w 22 ms | 1.00 (0.95) 0 ms / w 7 ms |

### sweep, chunk rows 65536

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | 7scenes | dtu | hiroom | nrgbd | scannetpp |
|---|---|---|---|---|---|
| none | 3.15 (1.05) 0 ms / w 1 ms | 3.15 (1.05) 0 ms / w 1 ms | 3.74 (1.04) 0 ms / w 1 ms | 3.15 (1.05) 0 ms / w 1 ms | 1.18 (1.12) 0 ms / w 0 ms |
| none+sh | 3.15 (1.05) 0 ms / w 1 ms | 3.15 (1.05) 0 ms / w 1 ms | 3.74 (1.04) 0 ms / w 1 ms | 3.15 (1.05) 0 ms / w 1 ms | 1.18 (1.12) 0 ms / w 0 ms |
| gzip1 | 2.49 (0.83) 1 ms / w 68 ms | 2.77 (0.92) 0 ms / w 55 ms | 2.32 (0.64) 0 ms / w 52 ms | 2.28 (0.76) 0 ms / w 53 ms | 0.87 (0.83) 0 ms / w 20 ms |
| gzip1+sh | 2.49 (0.83) 0 ms / w 57 ms | 2.77 (0.92) 0 ms / w 56 ms | 2.32 (0.64) 0 ms / w 55 ms | 2.28 (0.76) 0 ms / w 53 ms | 0.87 (0.83) 0 ms / w 19 ms |
| *gzip4 | 2.54 (0.85) 0 ms / w 68 ms | 2.83 (0.94) 0 ms / w 68 ms | 2.34 (0.65) 1 ms / w 93 ms | 2.49 (0.83) 0 ms / w 92 ms | 0.87 (0.83) 0 ms / w 22 ms |
| gzip4+sh | 2.54 (0.85) 0 ms / w 71 ms | 2.83 (0.94) 0 ms / w 66 ms | 2.34 (0.65) 0 ms / w 96 ms | 2.49 (0.83) 0 ms / w 91 ms | 0.87 (0.83) 0 ms / w 26 ms |
| gzip6 | 2.54 (0.85) 0 ms / w 75 ms | 2.83 (0.94) 0 ms / w 77 ms | 2.31 (0.64) 0 ms / w 148 ms | 2.49 (0.83) 0 ms / w 117 ms | 0.87 (0.83) 0 ms / w 24 ms |
| gzip6+sh | 2.54 (0.85) 0 ms / w 76 ms | 2.83 (0.94) 0 ms / w 80 ms | 2.31 (0.64) 0 ms / w 162 ms | 2.49 (0.83) 0 ms / w 119 ms | 0.87 (0.83) 0 ms / w 24 ms |
| gzip9 | 2.54 (0.85) 0 ms / w 80 ms | 2.83 (0.94) 0 ms / w 128 ms | 2.30 (0.64) 1 ms / w 620 ms | 2.49 (0.83) 0 ms / w 121 ms | 0.87 (0.83) 0 ms / w 30 ms |
| gzip9+sh | 2.54 (0.85) 0 ms / w 84 ms | 2.83 (0.94) 0 ms / w 143 ms | 2.30 (0.64) 1 ms / w 656 ms | 2.49 (0.83) 0 ms / w 119 ms | 0.87 (0.83) 0 ms / w 30 ms |
| lzf | 2.86 (0.95) 0 ms / w 30 ms | 2.98 (0.99) 0 ms / w 19 ms | 2.79 (0.77) 0 ms / w 30 ms | 2.76 (0.92) 0 ms / w 21 ms | 1.00 (0.95) 0 ms / w 6 ms |
| lzf+sh | 2.86 (0.95) 0 ms / w 28 ms | 2.98 (0.99) 0 ms / w 20 ms | 2.79 (0.77) 0 ms / w 30 ms | 2.76 (0.92) 0 ms / w 22 ms | 1.00 (0.95) 0 ms / w 6 ms |

### sweep, chunk rows 262144

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | 7scenes | dtu | hiroom | nrgbd | scannetpp |
|---|---|---|---|---|---|
| none | 3.15 (1.05) 0 ms / w 1 ms | 3.15 (1.05) 0 ms / w 1 ms | 3.93 (1.09) 0 ms / w 1 ms | 3.15 (1.05) 0 ms / w 1 ms | 1.57 (1.50) 0 ms / w 0 ms |
| none+sh | 3.15 (1.05) 0 ms / w 1 ms | 3.15 (1.05) 0 ms / w 0 ms | 3.93 (1.09) 0 ms / w 1 ms | 3.15 (1.05) 0 ms / w 1 ms | 1.57 (1.50) 0 ms / w 0 ms |
| gzip1 | 2.49 (0.83) 0 ms / w 64 ms | 2.76 (0.92) 0 ms / w 57 ms | 2.32 (0.64) 0 ms / w 55 ms | 2.26 (0.75) 0 ms / w 55 ms | 0.88 (0.83) 0 ms / w 20 ms |
| gzip1+sh | 2.49 (0.83) 0 ms / w 57 ms | 2.76 (0.92) 0 ms / w 57 ms | 2.32 (0.64) 0 ms / w 62 ms | 2.26 (0.75) 0 ms / w 54 ms | 0.88 (0.83) 0 ms / w 20 ms |
| *gzip4 | 2.54 (0.85) 0 ms / w 73 ms | 2.83 (0.94) 0 ms / w 69 ms | 2.33 (0.65) 0 ms / w 102 ms | 2.48 (0.83) 0 ms / w 95 ms | 0.87 (0.83) 0 ms / w 24 ms |
| gzip4+sh | 2.54 (0.85) 0 ms / w 73 ms | 2.83 (0.94) 0 ms / w 68 ms | 2.33 (0.65) 0 ms / w 99 ms | 2.48 (0.83) 0 ms / w 94 ms | 0.87 (0.83) 0 ms / w 30 ms |
| gzip6 | 2.54 (0.85) 0 ms / w 77 ms | 2.83 (0.94) 0 ms / w 78 ms | 2.31 (0.64) 0 ms / w 152 ms | 2.49 (0.83) 0 ms / w 123 ms | 0.87 (0.83) 0 ms / w 26 ms |
| gzip6+sh | 2.54 (0.85) 0 ms / w 82 ms | 2.83 (0.94) 0 ms / w 79 ms | 2.31 (0.64) 0 ms / w 158 ms | 2.49 (0.83) 0 ms / w 124 ms | 0.87 (0.83) 0 ms / w 26 ms |
| gzip9 | 2.54 (0.85) 0 ms / w 83 ms | 2.83 (0.94) 0 ms / w 140 ms | 2.29 (0.64) 0 ms / w 695 ms | 2.49 (0.83) 0 ms / w 126 ms | 0.87 (0.83) 0 ms / w 31 ms |
| gzip9+sh | 2.54 (0.85) 0 ms / w 101 ms | 2.83 (0.94) 0 ms / w 157 ms | 2.29 (0.64) 0 ms / w 683 ms | 2.49 (0.83) 0 ms / w 124 ms | 0.87 (0.83) 0 ms / w 31 ms |
| lzf | 2.86 (0.95) 0 ms / w 27 ms | 2.98 (0.99) 0 ms / w 17 ms | 2.79 (0.77) 0 ms / w 30 ms | 2.76 (0.92) 0 ms / w 20 ms | 1.01 (0.96) 0 ms / w 6 ms |
| lzf+sh | 2.86 (0.95) 1 ms / w 48 ms | 2.98 (0.99) 0 ms / w 19 ms | 2.79 (0.77) 0 ms / w 29 ms | 2.76 (0.92) 0 ms / w 20 ms | 1.01 (0.96) 0 ms / w 6 ms |

### policy — decision rule §5.0 (totals over the 5 sources above)

`*` = C1; min total = 10.70 MB.

| option | chunk | total MB | total read ms | lossless | ≤ 5 % | read ≤ 2× C4 | chosen |
|---|---|---|---|---|---|---|---|
| none | 16384 | 13.86 | 1 | y |  |  |  |
| none | 65536 | 14.35 | 1 | y |  |  |  |
| none | 262144 | 14.94 | 1 | y |  |  |  |
| none+sh | 16384 | 13.86 | 1 | y |  |  |  |
| none+sh | 65536 | 14.35 | 1 | y |  |  |  |
| none+sh | 262144 | 14.94 | 1 | y |  |  |  |
| gzip1 | 16384 | 10.88 | 2 | y | y | y |  |
| gzip1 | 65536 | 10.74 | 2 | y | y |  |  |
| gzip1 | 262144 | 10.70 | 1 | y | y | y |  |
| gzip1+sh | 16384 | 10.88 | 1 | y | y | y |  |
| gzip1+sh | 65536 | 10.74 | 1 | y | y | y |  |
| gzip1+sh | 262144 | 10.70 | 1 | y | y | y |  |
| *gzip4 | 16384 | 11.12 | 2 | y | y |  |  |
| *gzip4 | 65536 | 11.07 | 1 | y | y |  |  |
| *gzip4 | 262144 | 11.06 | 1 | y | y | y | **chosen** |
| gzip4+sh | 16384 | 11.12 | 2 | y | y |  |  |
| gzip4+sh | 65536 | 11.07 | 1 | y | y |  |  |
| gzip4+sh | 262144 | 11.06 | 1 | y | y | y |  |
| gzip6 | 16384 | 11.09 | 2 | y | y | y |  |
| gzip6 | 65536 | 11.04 | 1 | y | y | y |  |
| gzip6 | 262144 | 11.03 | 1 | y | y | y |  |
| gzip6+sh | 16384 | 11.09 | 2 | y | y | y |  |
| gzip6+sh | 65536 | 11.04 | 1 | y | y | y |  |
| gzip6+sh | 262144 | 11.03 | 1 | y | y | y |  |
| gzip9 | 16384 | 11.07 | 1 | y | y | y |  |
| gzip9 | 65536 | 11.03 | 1 | y | y |  |  |
| gzip9 | 262144 | 11.01 | 1 | y | y | y |  |
| gzip9+sh | 16384 | 11.07 | 2 | y | y |  |  |
| gzip9+sh | 65536 | 11.03 | 2 | y | y |  |  |
| gzip9+sh | 262144 | 11.01 | 1 | y | y | y |  |
| lzf | 16384 | 12.43 | 2 | y |  |  |  |
| lzf | 65536 | 12.40 | 1 | y |  |  |  |
| lzf | 262144 | 12.39 | 1 | y |  |  |  |
| lzf+sh | 16384 | 12.43 | 2 | y |  |  |  |
| lzf+sh | 65536 | 12.40 | 2 | y |  |  |  |
| lzf+sh | 262144 | 12.39 | 2 | y |  |  |  |

## N  (7scenes only (the only GT with normals))

### sweep, chunk rows 16384

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | 7scenes |
|---|---|
| none | 12.19 (1.02) 4 ms / w 5 ms |
| none+sh | 12.19 (1.02) 7 ms / w 9 ms |
| gzip1 | 11.19 (0.93) 54 ms / w 407 ms |
| gzip1+sh | 10.20 (0.85) 35 ms / w 267 ms |
| gzip4 | 11.15 (0.93) 43 ms / w 381 ms |
| *gzip4+sh | 10.09 (0.84) 24 ms / w 231 ms |
| gzip6 | 11.15 (0.93) 39 ms / w 333 ms |
| gzip6+sh | 10.02 (0.84) 24 ms / w 532 ms |
| gzip9 | 11.15 (0.93) 39 ms / w 336 ms |
| gzip9+sh | 10.01 (0.83) 23 ms / w 1204 ms |
| lzf | 12.00 (1.00) 6 ms / w 83 ms |
| lzf+sh | 11.09 (0.92) 21 ms / w 59 ms |

### sweep, chunk rows 65536

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | 7scenes |
|---|---|
| none | 12.58 (1.05) 3 ms / w 5 ms |
| none+sh | 12.58 (1.05) 6 ms / w 8 ms |
| gzip1 | 11.19 (0.93) 51 ms / w 413 ms |
| gzip1+sh | 10.19 (0.85) 33 ms / w 293 ms |
| gzip4 | 11.15 (0.93) 41 ms / w 385 ms |
| *gzip4+sh | 10.09 (0.84) 23 ms / w 232 ms |
| gzip6 | 11.15 (0.93) 38 ms / w 354 ms |
| gzip6+sh | 10.01 (0.83) 23 ms / w 577 ms |
| gzip9 | 11.15 (0.93) 38 ms / w 350 ms |
| gzip9+sh | 9.98 (0.83) 22 ms / w 1598 ms |
| lzf | 12.01 (1.00) 8 ms / w 77 ms |
| lzf+sh | 11.08 (0.92) 24 ms / w 63 ms |

### sweep, chunk rows 262144

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | 7scenes |
|---|---|
| none | 12.58 (1.05) 4 ms / w 4 ms |
| none+sh | 12.58 (1.05) 5 ms / w 8 ms |
| gzip1 | 11.19 (0.93) 42 ms / w 415 ms |
| gzip1+sh | 10.17 (0.85) 22 ms / w 264 ms |
| gzip4 | 11.15 (0.93) 32 ms / w 393 ms |
| *gzip4+sh | 10.08 (0.84) 18 ms / w 232 ms |
| gzip6 | 11.15 (0.93) 30 ms / w 348 ms |
| gzip6+sh | 10.01 (0.83) 18 ms / w 589 ms |
| gzip9 | 11.15 (0.93) 31 ms / w 353 ms |
| gzip9+sh | 9.97 (0.83) 19 ms / w 1715 ms |
| lzf | 12.07 (1.01) 3 ms / w 92 ms |
| lzf+sh | 11.08 (0.92) 20 ms / w 65 ms |

### policy — decision rule §5.0 (totals over the 1 sources above)

`*` = C1; min total = 9.97 MB.

| option | chunk | total MB | total read ms | lossless | ≤ 5 % | read ≤ 2× C4 | chosen |
|---|---|---|---|---|---|---|---|
| none | 16384 | 12.19 | 4 | y |  |  |  |
| none | 65536 | 12.58 | 3 | y |  |  |  |
| none | 262144 | 12.58 | 4 | y |  |  |  |
| none+sh | 16384 | 12.19 | 7 | y |  |  |  |
| none+sh | 65536 | 12.58 | 6 | y |  |  |  |
| none+sh | 262144 | 12.58 | 5 | y |  |  |  |
| gzip1 | 16384 | 11.19 | 54 | y |  |  |  |
| gzip1 | 65536 | 11.19 | 51 | y |  |  |  |
| gzip1 | 262144 | 11.19 | 42 | y |  |  |  |
| gzip1+sh | 16384 | 10.20 | 35 | y | y |  |  |
| gzip1+sh | 65536 | 10.19 | 33 | y | y |  |  |
| gzip1+sh | 262144 | 10.17 | 22 | y | y |  |  |
| gzip4 | 16384 | 11.15 | 43 | y |  |  |  |
| gzip4 | 65536 | 11.15 | 41 | y |  |  |  |
| gzip4 | 262144 | 11.15 | 32 | y |  |  |  |
| *gzip4+sh | 16384 | 10.09 | 24 | y | y |  |  |
| *gzip4+sh | 65536 | 10.09 | 23 | y | y |  |  |
| *gzip4+sh | 262144 | 10.08 | 18 | y | y |  |  |
| gzip6 | 16384 | 11.15 | 39 | y |  |  |  |
| gzip6 | 65536 | 11.15 | 38 | y |  |  |  |
| gzip6 | 262144 | 11.15 | 30 | y |  |  |  |
| gzip6+sh | 16384 | 10.02 | 24 | y | y |  |  |
| gzip6+sh | 65536 | 10.01 | 23 | y | y |  |  |
| gzip6+sh | 262144 | 10.01 | 18 | y | y |  | **chosen** |
| gzip9 | 16384 | 11.15 | 39 | y |  |  |  |
| gzip9 | 65536 | 11.15 | 38 | y |  |  |  |
| gzip9 | 262144 | 11.15 | 31 | y |  |  |  |
| gzip9+sh | 16384 | 10.01 | 23 | y | y |  |  |
| gzip9+sh | 65536 | 9.98 | 22 | y | y |  |  |
| gzip9+sh | 262144 | 9.97 | 19 | y | y |  |  |
| lzf | 16384 | 12.00 | 6 | y |  |  |  |
| lzf | 65536 | 12.01 | 8 | y |  |  |  |
| lzf | 262144 | 12.07 | 3 | y |  |  |  |
| lzf+sh | 16384 | 11.09 | 21 | y |  |  |  |
| lzf+sh | 65536 | 11.08 | 24 | y |  |  |  |
| lzf+sh | 262144 | 11.08 | 20 | y |  |  |  |

## ptnum  (scannetpp only (the only GT with faces))

### sweep, chunk rows 16384

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | scannetpp |
|---|---|
| none | 8.59 (1.01) 1 ms / w 2 ms |
| none+sh | 8.59 (1.01) 3 ms / w 4 ms |
| gzip1 | 3.34 (0.39) 27 ms / w 82 ms |
| gzip1+sh | 3.20 (0.38) 19 ms / w 58 ms |
| gzip4 | 3.42 (0.40) 27 ms / w 126 ms |
| *gzip4+sh | 3.11 (0.36) 23 ms / w 82 ms |
| gzip6 | 3.42 (0.40) 27 ms / w 488 ms |
| gzip6+sh | 3.10 (0.36) 23 ms / w 169 ms |
| gzip9 | 3.42 (0.40) 28 ms / w 1225 ms |
| gzip9+sh | 3.09 (0.36) 31 ms / w 615 ms |
| lzf | 5.81 (0.68) 26 ms / w 61 ms |
| lzf+sh | 3.75 (0.44) 24 ms / w 55 ms |

### sweep, chunk rows 65536

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | scannetpp |
|---|---|
| none | 8.65 (1.01) 1 ms / w 2 ms |
| none+sh | 8.65 (1.01) 1 ms / w 4 ms |
| gzip1 | 3.34 (0.39) 3 ms / w 81 ms |
| gzip1+sh | 3.20 (0.38) 2 ms / w 70 ms |
| gzip4 | 3.41 (0.40) 3 ms / w 124 ms |
| *gzip4+sh | 3.11 (0.37) 2 ms / w 95 ms |
| gzip6 | 3.42 (0.40) 3 ms / w 522 ms |
| gzip6+sh | 3.11 (0.36) 2 ms / w 191 ms |
| gzip9 | 3.40 (0.40) 3 ms / w 1540 ms |
| gzip9+sh | 3.10 (0.36) 5 ms / w 747 ms |
| lzf | 5.79 (0.68) 4 ms / w 59 ms |
| lzf+sh | 3.74 (0.44) 4 ms / w 46 ms |

### sweep, chunk rows 262144

cells: stored MB (ratio to raw) · read ms / write ms; raw is the f32/u8/i32 array the H5Geo writers will store.

| option | scannetpp |
|---|---|
| none | 9.44 (1.11) 1 ms / w 1 ms |
| none+sh | 9.44 (1.11) 3 ms / w 4 ms |
| gzip1 | 3.34 (0.39) 26 ms / w 81 ms |
| gzip1+sh | 3.18 (0.37) 18 ms / w 74 ms |
| gzip4 | 3.41 (0.40) 24 ms / w 125 ms |
| *gzip4+sh | 3.11 (0.36) 20 ms / w 100 ms |
| gzip6 | 3.42 (0.40) 24 ms / w 526 ms |
| gzip6+sh | 3.11 (0.36) 19 ms / w 199 ms |
| gzip9 | 3.40 (0.40) 29 ms / w 1548 ms |
| gzip9+sh | 3.10 (0.36) 26 ms / w 765 ms |
| lzf | 5.79 (0.68) 26 ms / w 58 ms |
| lzf+sh | 3.74 (0.44) 23 ms / w 49 ms |

### policy — decision rule §5.0 (totals over the 1 sources above)

`*` = C1; min total = 3.09 MB.

| option | chunk | total MB | total read ms | lossless | ≤ 5 % | read ≤ 2× C4 | chosen |
|---|---|---|---|---|---|---|---|
| none | 16384 | 8.59 | 1 | y |  |  |  |
| none | 65536 | 8.65 | 1 | y |  |  |  |
| none | 262144 | 9.44 | 1 | y |  |  |  |
| none+sh | 16384 | 8.59 | 3 | y |  |  |  |
| none+sh | 65536 | 8.65 | 1 | y |  |  |  |
| none+sh | 262144 | 9.44 | 3 | y |  |  |  |
| gzip1 | 16384 | 3.34 | 27 | y |  |  |  |
| gzip1 | 65536 | 3.34 | 3 | y |  |  |  |
| gzip1 | 262144 | 3.34 | 26 | y |  |  |  |
| gzip1+sh | 16384 | 3.20 | 19 | y | y |  |  |
| gzip1+sh | 65536 | 3.20 | 2 | y | y |  | **chosen** |
| gzip1+sh | 262144 | 3.18 | 18 | y | y |  |  |
| gzip4 | 16384 | 3.42 | 27 | y |  |  |  |
| gzip4 | 65536 | 3.41 | 3 | y |  |  |  |
| gzip4 | 262144 | 3.41 | 24 | y |  |  |  |
| *gzip4+sh | 16384 | 3.11 | 23 | y | y |  |  |
| *gzip4+sh | 65536 | 3.11 | 2 | y | y |  |  |
| *gzip4+sh | 262144 | 3.11 | 20 | y | y |  |  |
| gzip6 | 16384 | 3.42 | 27 | y |  |  |  |
| gzip6 | 65536 | 3.42 | 3 | y |  |  |  |
| gzip6 | 262144 | 3.42 | 24 | y |  |  |  |
| gzip6+sh | 16384 | 3.10 | 23 | y | y |  |  |
| gzip6+sh | 65536 | 3.11 | 2 | y | y |  |  |
| gzip6+sh | 262144 | 3.11 | 19 | y | y |  |  |
| gzip9 | 16384 | 3.42 | 28 | y |  |  |  |
| gzip9 | 65536 | 3.40 | 3 | y |  |  |  |
| gzip9 | 262144 | 3.40 | 29 | y |  |  |  |
| gzip9+sh | 16384 | 3.09 | 31 | y | y |  |  |
| gzip9+sh | 65536 | 3.10 | 5 | y | y |  |  |
| gzip9+sh | 262144 | 3.10 | 26 | y | y |  |  |
| lzf | 16384 | 5.81 | 26 | y |  |  |  |
| lzf | 65536 | 5.79 | 4 | y |  |  |  |
| lzf | 262144 | 5.79 | 26 | y |  |  |  |
| lzf+sh | 16384 | 3.75 | 24 | y |  |  |  |
| lzf+sh | 65536 | 3.74 | 4 | y |  |  |  |
| lzf+sh | 262144 | 3.74 | 23 | y |  |  |  |

## ptnum layout (C6): flat (V,) vs (F,3)

Same byte content, two layouts. Stored MB at the chosen filter and at raw:

| filter / chunk | flat (V,) | (F,3) |
|---|---|---|
| gzip1+sh × 65536 | 3.20 | 3.18 |
| none × 65536 | 8.65 | 8.65 |

Verdict: flat 3.20 MB vs (F,3) 3.18 MB — they differ; adopt the smaller.

## per-source h5/ply ratio and 43-scene projection

Per source: h5 bytes = sum of the chosen-policy cells for the attributes that source has (P always; Cd; N for 7scenes; ptnum for scannetpp). Ply bytes = the sample file on disk; the 43-scene total per source is from `REQUIREMENTS_GT_GEOMETRY.md` (~1.8 GB; scannetpp = the remainder, estimate).

| source | sample ply MB | h5 MB | ratio | 43-scene total MB | projected MB |
|---|---|---|---|---|---|
| 7scenes | 51.0 | 20.73 | 0.41 | 357 | 145 |
| dtu | 27.0 | 12.65 | 0.47 | 351 | 164 |
| hiroom | 32.5 | 13.46 | 0.41 | 228 | 94 |
| nrgbd | 27.0 | 12.45 | 0.46 | 216 | 100 |
| scannetpp | 14.5 | 6.88 | 0.47 | 648 | 308 |
| **total** |  |  |  | 1800 | **811** |

Global check (§5.0 rule 6): 811 MB < 1.8 GB, pass (expected ≈ 0.8 GB, measured 0.81 GB).

## C7 per-source exception (plain gzip for grid-like P)

7scenes P lies on an f32-exact grid, where shuffle (which breaks long identical runs) hurts. At the chosen chunk rows 262144: chosen-policy cell 8.18 MB, plain gzip4 6.77 MB, plain gzip9 6.72 MB. Projected saving **10.2 MB** vs the 10 %-of-projected-total threshold **81.1 MB** → **not worth it**, keep one global policy.

## C8 info-only (ineligible: lossy or a new dependency)

| option | stored MB | max abs err | note |
|---|---|---|---|
| N as f16 (7scenes) | 4.96 | 2.4e-04 | lossy, ineligible |
| P quantised to 1e-3 m grid (7scenes) | 7.80 | 5.0e-04 | lossy, ineligible |
| P quantised to 1e-3 m grid (dtu) | 8.50 | 5.0e-04 | lossy, ineligible |
| P quantised to 1e-3 m grid (hiroom) | 9.77 | 5.0e-04 | lossy, ineligible |
| P quantised to 1e-3 m grid (nrgbd) | 9.23 | 5.0e-04 | lossy, ineligible |
| P quantised to 1e-3 m grid (scannetpp) | 2.72 | 5.0e-04 | lossy, ineligible |
| zstd via hdf5plugin | — | — | hdf5plugin NOT installed (zstd not measured; new dependency, ineligible) |

## probes (50k points, §5.3)

Written once by `pixi run smoke-geo --write-probes`; no ply parser needed at test time. scannetpp: the first 50k points plus the submesh whose faces use only those points (re-indexed to identity).

| source | P.npy | Cd.npy | N.npy | ptnum.npy |
|---|---|---|---|---|
| 7scenes | (50000, 3) float32 | (50000, 3) uint8 | (50000, 3) float32 | — |
| dtu | (50000, 3) float32 | (50000, 3) uint8 | — | — |
| hiroom | (50000, 3) float32 | (50000, 3) uint8 | — | — |
| nrgbd | (50000, 3) float32 | (50000, 3) uint8 | — | — |
| scannetpp | (50000, 3) float32 | (50000, 3) uint8 | — | (283428,) int32 |


## §5.4 full-set acceptance (SpatialBench side, 2026-11-13 15:32 NZDT)

Converter: `SpatialBench/_preprocess/convert_pointcloud_to_h5.py` (open3d 0.18; D7),
per-scene acceptance in-place. All 45 scenes (the requirement doc counted 43;
actual 7scenes 7 + dtu 13 + hiroom 6 + nrgbd 8 + scannetpp 11 = 45) pass.

- total 0.711 GB h5 vs 1.922 GB ply (< 1.8 GB pass; below the 0.81 GB projection)
- conversion 50.3 s for the whole set (per-scene 0.1–2.8 s)
- full-read (P, + Cd/N/ptnum): min 0.025 s / median 0.069 s / max 0.347 s
- P max abs diff 2.38e-07 m (all scenes; f64 sources stored f32; atol 1e-06 pass)
- N max abs diff 0.0 (7scenes only; f64 source values are f32-exact)
- Cd bit-exact, ptnum bit-exact (all 11 scannetpp meshes)
- mesh sample counts: original vs h5-rebuilt `sample_points_uniformly(1_000_000)`
  equal on all 11 meshes (1 000 000 == 1 000 000)

Per source (h5 / ply bytes): 7scenes 119 MB / 332 MB (with N) | dtu 134 MB / 364 MB
| hiroom 34 MB / 99 MB | nrgbd 81 MB / 216 MB | scannetpp 342 MB / 911 MB (with ptnum).
