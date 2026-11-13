"""Cut small REAL samples out of the vggt_research data volume (old flat h5).

Each sample keeps the old flat layout byte-for-byte for the frames it keeps
(rgb bytes untouched, depth/mask/K/T rows, every /meta key sliced per frame),
so it is a faithful input for a converter and for the E1/E2 acceptance tests.

Run from the vggt_research root (needs the data volume symlink `dataset/`):
  cd /vol/juneleung-solar/lizard/vggt_research
  pixi run python ../libs-dev/h5scene/requirements_eval/samples/build_samples.py
"""
import os.path as osp

import h5py
import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
# (source h5 under dataset/, out name, frame ids to keep, why)
CUTS = [
    ("7scenes/pumpkin/seq-01.h5", "7scenes_pumpkin_seq-01_cut5.h5",
     ["000000", "000100", "000200", "000300", "000400"],
     "fixed-size baseline; int-like fids; per-frame split 'test'"),
    ("tum/walking_xyz.h5", "tum_walking_xyz_cut5.h5", None,
     "fids are TIMESTAMP strings ('1341846313.553992'): src_frame_idxs cannot hold them"),
    ("co3d/car/564_81552_160806/train.h5", "co3d_car_564_cut6.h5",
     ["000001", "000012", "000014", "000051", "000002", "000022"],
     "VARIABLE per-frame size: heights 696/703/700/698 x 384 (E2)"),
]


def cut(src, dst, keep):
    with h5py.File(src, "r") as h, h5py.File(dst, "w") as o:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in h["meta/images"][:]]
        if keep is None:
            keep = ids[::max(1, len(ids) // 5)][:5]
        pos = [ids.index(f) for f in keep]
        n = len(ids)
        for k, v in h.attrs.items():
            o.attrs[k] = v
        o.attrs["nframes"] = len(keep)
        o.attrs["_cut_from"] = src
        g = o.create_group("rgb")
        for f in keep:
            g.create_dataset(f, data=h["rgb"][f][()])
        for name in ("depth", "mask", "depth_conf"):
            if name not in h:
                continue
            if isinstance(h[name], h5py.Group):            # per-frame (variable size)
                gg = o.create_group(name)
                for f in keep:
                    if f in h[name]:
                        gg.create_dataset(f, data=h[name][f][()], compression="gzip")
            else:
                o.create_dataset(name, data=h[name][pos], compression="gzip")
        for name in ("K", "T"):
            if name in h:
                o.create_dataset(name, data=h[name][pos])
        m = o.create_group("meta")
        for k, v in h["meta"].items():
            if isinstance(v, h5py.Dataset) and v.shape == (n,):
                m.create_dataset(k, data=v[pos])
            elif isinstance(v, h5py.Dataset):
                m.create_dataset(k, data=v[()])
        # groups / group_scores (training CSR) are dropped: not an eval need
    return keep


def main():
    lines = ["# samples (cut from the vggt_research data volume, old flat layout)", ""]
    for src, name, keep, why in CUTS:
        keep = cut(osp.join("dataset", src), osp.join(HERE, name), keep)
        lines.append(f"- `{name}` <- `dataset/{src}` frames {keep}: {why}")
        print(name, keep)
    with open(osp.join(HERE, "SAMPLES.md"), "w") as f:
        f.write("\n".join(lines) + "\n\nOld flat layout: vggt_research "
                "`dataset/_preprocess/README.md` §0.1 (copy in ../reference/).\n")


if __name__ == "__main__":
    main()
