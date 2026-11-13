# hdf5plugin (zstd / blosc / lz4) compression test

Run 2026-11-13 15:05 NZDT via `pixi run smoke-plugin` (pixi `plugin` env, hdf5plugin 7.x, h5py 3.16.0). Same protocol as `tests/probes/geo/README.md`: stored MB, ratio to raw, write time omitted here (not decisive), full-read time = `ds[:]` (the consumer's only access pattern), bit-exact round trip. Chunks: 16384 rows; depth = full 640x640 frame.

## geo/7scenes P  (raw 0.60 MB, dtype float32)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.786 | 1.311 | 0.04 | 1.87x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.421 | 0.701 | 0.04 | 1.00x | 1.04x | ok |
| zstd1 | 0.416 | 0.694 | 0.04 | 0.99x | 0.98x | ok |
| zstd3 | 0.345 | 0.575 | 0.04 | 0.82x | 0.96x | ok |
| zstd9 | 0.333 | 0.554 | 0.03 | 0.79x | 0.94x | ok |
| blosc2-lz4 c5 | 0.575 | 0.958 | 0.03 | 1.37x | 0.94x | ok |
| blosc2-lz4 c5 shuf | 0.518 | 0.864 | 0.03 | 1.23x | 0.94x | ok |
| blosc2-zstd c3 | 0.341 | 0.568 | 0.04 | 0.81x | 0.95x | ok |
| blosc2-zlib c5 | 0.343 | 0.571 | 0.04 | 0.81x | 0.97x | ok |
| blosc1-lz4 c5 shuf | 0.517 | 0.862 | 0.03 | 1.23x | 0.95x | ok |
| blosc1-lz4 c5 noshuf | 0.546 | 0.909 | 0.03 | 1.30x | 0.93x | ok |
| lz4 (hdf5plugin) | 0.509 | 0.848 | 0.03 | 1.21x | 0.94x | ok |

## geo/7scenes Cd  (raw 0.15 MB, dtype uint8)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.197 | 1.311 | 0.01 | 1.53x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.129 | 0.859 | 0.01 | 1.00x | 0.99x | ok |
| zstd1 | 0.138 | 0.917 | 0.01 | 1.07x | 0.98x | ok |
| zstd3 | 0.134 | 0.891 | 0.01 | 1.04x | 0.97x | ok |
| zstd9 | 0.131 | 0.874 | 0.01 | 1.02x | 0.96x | ok |
| blosc2-lz4 c5 | 0.150 | 1.003 | 0.01 | 1.17x | 0.97x | ok |
| blosc2-lz4 c5 shuf | 0.150 | 1.003 | 0.01 | 1.17x | 0.97x | ok |
| blosc2-zstd c3 | 0.134 | 0.892 | 0.01 | 1.04x | 0.98x | ok |
| blosc2-zlib c5 | 0.130 | 0.866 | 0.01 | 1.01x | 0.98x | ok |
| blosc1-lz4 c5 shuf | 0.149 | 0.996 | 0.01 | 1.16x | 0.97x | ok |
| blosc1-lz4 c5 noshuf | 0.149 | 0.996 | 0.01 | 1.16x | 0.95x | ok |
| lz4 (hdf5plugin) | 0.146 | 0.975 | 0.01 | 1.14x | 0.93x | ok |

## geo/7scenes N  (raw 0.60 MB, dtype float32)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.786 | 1.311 | 0.04 | 1.55x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.506 | 0.843 | 0.04 | 1.00x | 0.97x | ok |
| zstd1 | 0.556 | 0.927 | 0.03 | 1.10x | 0.96x | ok |
| zstd3 | 0.556 | 0.927 | 0.03 | 1.10x | 0.97x | ok |
| zstd9 | 0.556 | 0.927 | 0.03 | 1.10x | 0.95x | ok |
| blosc2-lz4 c5 | 0.602 | 1.003 | 0.03 | 1.19x | 0.95x | ok |
| blosc2-lz4 c5 shuf | 0.553 | 0.922 | 0.03 | 1.09x | 0.97x | ok |
| blosc2-zstd c3 | 0.557 | 0.929 | 0.03 | 1.10x | 0.96x | ok |
| blosc2-zlib c5 | 0.559 | 0.932 | 0.03 | 1.11x | 0.97x | ok |
| blosc1-lz4 c5 shuf | 0.552 | 0.920 | 0.03 | 1.09x | 0.96x | ok |
| blosc1-lz4 c5 noshuf | 0.601 | 1.001 | 0.03 | 1.19x | 0.97x | ok |
| lz4 (hdf5plugin) | 0.601 | 1.001 | 0.03 | 1.19x | 0.96x | ok |

## geo/dtu P  (raw 0.60 MB, dtype float32)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.786 | 1.311 | 0.03 | 1.58x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.497 | 0.828 | 0.03 | 1.00x | 0.99x | ok |
| zstd1 | 0.553 | 0.922 | 0.03 | 1.11x | 1.00x | ok |
| zstd3 | 0.553 | 0.922 | 0.03 | 1.11x | 0.98x | ok |
| zstd9 | 0.553 | 0.922 | 0.03 | 1.11x | 0.98x | ok |
| blosc2-lz4 c5 | 0.602 | 1.003 | 0.03 | 1.21x | 0.99x | ok |
| blosc2-lz4 c5 shuf | 0.544 | 0.906 | 0.03 | 1.09x | 0.98x | ok |
| blosc2-zstd c3 | 0.554 | 0.923 | 0.03 | 1.11x | 0.99x | ok |
| blosc2-zlib c5 | 0.556 | 0.927 | 0.03 | 1.12x | 0.99x | ok |
| blosc1-lz4 c5 shuf | 0.543 | 0.905 | 0.03 | 1.09x | 0.99x | ok |
| blosc1-lz4 c5 noshuf | 0.601 | 1.001 | 0.03 | 1.21x | 0.99x | ok |
| lz4 (hdf5plugin) | 0.601 | 1.001 | 0.03 | 1.21x | 0.98x | ok |

## geo/dtu Cd  (raw 0.15 MB, dtype uint8)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.197 | 1.311 | 0.01 | 1.38x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.142 | 0.947 | 0.01 | 1.00x | 0.99x | ok |
| zstd1 | 0.144 | 0.958 | 0.01 | 1.01x | 0.97x | ok |
| zstd3 | 0.144 | 0.958 | 0.01 | 1.01x | 0.99x | ok |
| zstd9 | 0.143 | 0.952 | 0.01 | 1.01x | 0.99x | ok |
| blosc2-lz4 c5 | 0.151 | 1.009 | 0.01 | 1.07x | 1.00x | ok |
| blosc2-lz4 c5 shuf | 0.151 | 1.009 | 0.01 | 1.07x | 1.00x | ok |
| blosc2-zstd c3 | 0.145 | 0.965 | 0.01 | 1.02x | 0.99x | ok |
| blosc2-zlib c5 | 0.143 | 0.954 | 0.01 | 1.01x | 1.00x | ok |
| blosc1-lz4 c5 shuf | 0.150 | 1.002 | 0.01 | 1.06x | 0.98x | ok |
| blosc1-lz4 c5 noshuf | 0.150 | 1.002 | 0.01 | 1.06x | 0.98x | ok |
| lz4 (hdf5plugin) | 0.150 | 1.000 | 0.01 | 1.06x | 0.98x | ok |

## geo/hiroom P  (raw 0.60 MB, dtype float32)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.786 | 1.311 | 0.04 | 1.77x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.443 | 0.739 | 0.04 | 1.00x | 1.01x | ok |
| zstd1 | 0.545 | 0.908 | 0.03 | 1.23x | 0.99x | ok |
| zstd3 | 0.545 | 0.908 | 0.03 | 1.23x | 0.98x | ok |
| zstd9 | 0.543 | 0.905 | 0.03 | 1.23x | 0.97x | ok |
| blosc2-lz4 c5 | 0.602 | 1.003 | 0.03 | 1.36x | 0.98x | ok |
| blosc2-lz4 c5 shuf | 0.492 | 0.820 | 0.03 | 1.11x | 0.99x | ok |
| blosc2-zstd c3 | 0.543 | 0.905 | 0.03 | 1.23x | 0.99x | ok |
| blosc2-zlib c5 | 0.524 | 0.873 | 0.04 | 1.18x | 1.06x | ok |
| blosc1-lz4 c5 shuf | 0.491 | 0.818 | 0.04 | 1.11x | 1.05x | ok |
| blosc1-lz4 c5 noshuf | 0.601 | 1.001 | 0.04 | 1.36x | 1.05x | ok |
| lz4 (hdf5plugin) | 0.601 | 1.001 | 0.04 | 1.36x | 1.06x | ok |

## geo/hiroom Cd  (raw 0.15 MB, dtype uint8)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.197 | 1.311 | 0.01 | 1.79x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.110 | 0.733 | 0.01 | 1.00x | 1.04x | ok |
| zstd1 | 0.125 | 0.832 | 0.01 | 1.14x | 0.98x | ok |
| zstd3 | 0.123 | 0.817 | 0.02 | 1.11x | 1.11x | ok |
| zstd9 | 0.118 | 0.787 | 0.01 | 1.07x | 0.99x | ok |
| blosc2-lz4 c5 | 0.141 | 0.939 | 0.01 | 1.28x | 0.99x | ok |
| blosc2-lz4 c5 shuf | 0.141 | 0.939 | 0.01 | 1.28x | 0.97x | ok |
| blosc2-zstd c3 | 0.119 | 0.796 | 0.01 | 1.09x | 1.01x | ok |
| blosc2-zlib c5 | 0.111 | 0.738 | 0.02 | 1.01x | 1.11x | ok |
| blosc1-lz4 c5 shuf | 0.140 | 0.932 | 0.01 | 1.27x | 1.06x | ok |
| blosc1-lz4 c5 noshuf | 0.140 | 0.932 | 0.01 | 1.27x | 1.04x | ok |
| lz4 (hdf5plugin) | 0.133 | 0.888 | 0.01 | 1.21x | 1.03x | ok |

## geo/nrgbd P  (raw 0.60 MB, dtype float32)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.786 | 1.311 | 0.04 | 1.56x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.503 | 0.839 | 0.05 | 1.00x | 1.09x | ok |
| zstd1 | 0.549 | 0.916 | 0.04 | 1.09x | 0.99x | ok |
| zstd3 | 0.549 | 0.916 | 0.04 | 1.09x | 0.98x | ok |
| zstd9 | 0.549 | 0.916 | 0.04 | 1.09x | 0.94x | ok |
| blosc2-lz4 c5 | 0.602 | 1.003 | 0.04 | 1.20x | 0.93x | ok |
| blosc2-lz4 c5 shuf | 0.552 | 0.920 | 0.04 | 1.10x | 0.92x | ok |
| blosc2-zstd c3 | 0.551 | 0.918 | 0.04 | 1.09x | 0.91x | ok |
| blosc2-zlib c5 | 0.552 | 0.921 | 0.05 | 1.10x | 1.22x | ok |
| blosc1-lz4 c5 shuf | 0.551 | 0.918 | 0.05 | 1.09x | 1.08x | ok |
| blosc1-lz4 c5 noshuf | 0.601 | 1.001 | 0.04 | 1.19x | 0.90x | ok |
| lz4 (hdf5plugin) | 0.601 | 1.001 | 0.04 | 1.19x | 0.89x | ok |

## geo/nrgbd Cd  (raw 0.15 MB, dtype uint8)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.197 | 1.311 | 0.01 | 1.57x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.125 | 0.834 | 0.02 | 1.00x | 1.06x | ok |
| zstd1 | 0.136 | 0.908 | 0.01 | 1.09x | 0.88x | ok |
| zstd3 | 0.136 | 0.908 | 0.01 | 1.09x | 0.92x | ok |
| zstd9 | 0.133 | 0.890 | 0.02 | 1.07x | 1.05x | ok |
| blosc2-lz4 c5 | 0.151 | 1.009 | 0.01 | 1.21x | 1.01x | ok |
| blosc2-lz4 c5 shuf | 0.151 | 1.009 | 0.01 | 1.21x | 1.02x | ok |
| blosc2-zstd c3 | 0.136 | 0.908 | 0.01 | 1.09x | 0.99x | ok |
| blosc2-zlib c5 | 0.126 | 0.841 | 0.02 | 1.01x | 1.05x | ok |
| blosc1-lz4 c5 shuf | 0.150 | 1.002 | 0.01 | 1.20x | 0.97x | ok |
| blosc1-lz4 c5 noshuf | 0.150 | 1.002 | 0.01 | 1.20x | 1.02x | ok |
| lz4 (hdf5plugin) | 0.150 | 1.001 | 0.01 | 1.20x | 0.95x | ok |

## geo/scannetpp P  (raw 0.60 MB, dtype float32)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.786 | 1.311 | 0.05 | 1.98x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.396 | 0.661 | 0.04 | 1.00x | 0.90x | ok |
| zstd1 | 0.484 | 0.807 | 0.04 | 1.22x | 0.82x | ok |
| zstd3 | 0.476 | 0.793 | 0.04 | 1.20x | 0.82x | ok |
| zstd9 | 0.472 | 0.786 | 0.04 | 1.19x | 0.86x | ok |
| blosc2-lz4 c5 | 0.600 | 1.001 | 0.04 | 1.51x | 0.87x | ok |
| blosc2-lz4 c5 shuf | 0.466 | 0.776 | 0.04 | 1.17x | 0.86x | ok |
| blosc2-zstd c3 | 0.475 | 0.791 | 0.04 | 1.20x | 0.82x | ok |
| blosc2-zlib c5 | 0.467 | 0.778 | 0.04 | 1.18x | 0.87x | ok |
| blosc1-lz4 c5 shuf | 0.465 | 0.774 | 0.04 | 1.17x | 0.80x | ok |
| blosc1-lz4 c5 noshuf | 0.595 | 0.991 | 0.04 | 1.50x | 0.82x | ok |
| lz4 (hdf5plugin) | 0.590 | 0.984 | 0.04 | 1.49x | 0.79x | ok |

## geo/scannetpp Cd  (raw 0.15 MB, dtype uint8)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.197 | 1.311 | 0.01 | 1.53x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.129 | 0.858 | 0.01 | 1.00x | 0.99x | ok |
| zstd1 | 0.137 | 0.916 | 0.01 | 1.07x | 0.98x | ok |
| zstd3 | 0.137 | 0.915 | 0.01 | 1.07x | 0.97x | ok |
| zstd9 | 0.136 | 0.909 | 0.01 | 1.06x | 0.99x | ok |
| blosc2-lz4 c5 | 0.151 | 1.008 | 0.01 | 1.17x | 0.99x | ok |
| blosc2-lz4 c5 shuf | 0.151 | 1.008 | 0.01 | 1.17x | 0.98x | ok |
| blosc2-zstd c3 | 0.135 | 0.897 | 0.01 | 1.04x | 0.96x | ok |
| blosc2-zlib c5 | 0.130 | 0.866 | 0.01 | 1.01x | 0.99x | ok |
| blosc1-lz4 c5 shuf | 0.150 | 1.001 | 0.01 | 1.17x | 1.00x | ok |
| blosc1-lz4 c5 noshuf | 0.150 | 1.001 | 0.01 | 1.17x | 0.98x | ok |
| lz4 (hdf5plugin) | 0.150 | 0.999 | 0.01 | 1.16x | 0.97x | ok |

## geo/scannetpp ptnum  (raw 1.13 MB, dtype int32)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 1.180 | 1.041 | 0.09 | 3.02x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.391 | 0.345 | 0.07 | 1.00x | 0.79x | ok |
| zstd1 | 0.506 | 0.446 | 0.07 | 1.29x | 0.79x | ok |
| zstd3 | 0.509 | 0.449 | 0.08 | 1.30x | 0.85x | ok |
| zstd9 | 0.471 | 0.415 | 0.08 | 1.20x | 0.85x | ok |
| blosc2-lz4 c5 | 0.901 | 0.795 | 0.08 | 2.30x | 0.85x | ok |
| blosc2-lz4 c5 shuf | 0.494 | 0.436 | 0.08 | 1.26x | 0.85x | ok |
| blosc2-zstd c3 | 0.478 | 0.422 | 0.08 | 1.22x | 0.83x | ok |
| blosc2-zlib c5 | 0.464 | 0.410 | 0.09 | 1.19x | 0.97x | ok |
| blosc1-lz4 c5 shuf | 0.494 | 0.436 | 0.08 | 1.26x | 0.81x | ok |
| blosc1-lz4 c5 noshuf | 0.949 | 0.837 | 0.07 | 2.43x | 0.81x | ok |
| lz4 (hdf5plugin) | 0.799 | 0.705 | 0.08 | 2.04x | 0.82x | ok |

## depth frame6 (uint16 640x640)  (raw 0.82 MB, dtype uint16)

| filter | stored MB | ratio | read ms | vs gzip4+sh (stored) | vs raw (read) | round trip |
|---|---|---|---|---|---|---|
| none | 0.819 | 1.000 | 0.05 | 3.14x | 1.00x | ok |
| gzip4+sh (h5py baseline) | 0.261 | 0.319 | 0.05 | 1.00x | 0.91x | ok |
| zstd1 | 0.365 | 0.446 | 0.05 | 1.40x | 0.94x | ok |
| zstd3 | 0.310 | 0.379 | 0.05 | 1.19x | 0.92x | ok |
| zstd9 | 0.289 | 0.352 | 0.05 | 1.10x | 0.86x | ok |
| blosc2-lz4 c5 | 0.534 | 0.652 | 0.05 | 2.04x | 0.85x | ok |
| blosc2-lz4 c5 shuf | 0.358 | 0.437 | 0.05 | 1.37x | 0.92x | ok |
| blosc2-zstd c3 | 0.317 | 0.387 | 0.05 | 1.21x | 0.99x | ok |
| blosc2-zlib c5 | 0.325 | 0.397 | 0.05 | 1.24x | 0.94x | ok |
| blosc1-lz4 c5 shuf | 0.352 | 0.429 | 0.05 | 1.35x | 0.91x | ok |
| blosc1-lz4 c5 noshuf | 0.562 | 0.686 | 0.05 | 2.15x | 0.92x | ok |
| lz4 (hdf5plugin) | 0.487 | 0.594 | 0.05 | 1.86x | 0.93x | ok |

## reading the numbers

Per-attribute one-line verdict (written after the table):

- geo/7scenes P: smallest plugin cell 0.333 MB (0.79x the gzip4+sh cell 0.421 MB), read 0.03 ms vs gzip4+sh 0.04 ms.
- geo/7scenes Cd: smallest plugin cell 0.130 MB (1.01x the gzip4+sh cell 0.129 MB), read 0.01 ms vs gzip4+sh 0.01 ms.
- geo/7scenes N: smallest plugin cell 0.552 MB (1.09x the gzip4+sh cell 0.506 MB), read 0.03 ms vs gzip4+sh 0.04 ms.
- geo/dtu P: smallest plugin cell 0.543 MB (1.09x the gzip4+sh cell 0.497 MB), read 0.03 ms vs gzip4+sh 0.03 ms.
- geo/dtu Cd: smallest plugin cell 0.143 MB (1.01x the gzip4+sh cell 0.142 MB), read 0.01 ms vs gzip4+sh 0.01 ms.
- geo/hiroom P: smallest plugin cell 0.491 MB (1.11x the gzip4+sh cell 0.443 MB), read 0.04 ms vs gzip4+sh 0.04 ms.
- geo/hiroom Cd: smallest plugin cell 0.111 MB (1.01x the gzip4+sh cell 0.110 MB), read 0.02 ms vs gzip4+sh 0.01 ms.
- geo/nrgbd P: smallest plugin cell 0.549 MB (1.09x the gzip4+sh cell 0.503 MB), read 0.04 ms vs gzip4+sh 0.05 ms.
- geo/nrgbd Cd: smallest plugin cell 0.126 MB (1.01x the gzip4+sh cell 0.125 MB), read 0.02 ms vs gzip4+sh 0.02 ms.
- geo/scannetpp P: smallest plugin cell 0.465 MB (1.17x the gzip4+sh cell 0.396 MB), read 0.04 ms vs gzip4+sh 0.04 ms.
- geo/scannetpp Cd: smallest plugin cell 0.130 MB (1.01x the gzip4+sh cell 0.129 MB), read 0.01 ms vs gzip4+sh 0.01 ms.
- geo/scannetpp ptnum: smallest plugin cell 0.464 MB (1.19x the gzip4+sh cell 0.391 MB), read 0.09 ms vs gzip4+sh 0.07 ms.
- depth frame6 (uint16 640x640): smallest plugin cell 0.289 MB (1.10x the gzip4+sh cell 0.261 MB), read 0.05 ms vs gzip4+sh 0.05 ms.

