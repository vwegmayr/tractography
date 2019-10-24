import argparse
import glob
import os

import numpy as np
import nibabel as nib

from nibabel.streamlines.tractogram import Tractogram
from nibabel.streamlines.trk import TrkFile


def merge_trks(trk_dir, keep, weighted, out_dir):
    """
    WARNING: Alignment between trk files is not checked, but assumed the same!
    """
    bundles = []
    for i, trk_path in enumerate(glob.glob(os.path.join(trk_dir, "*.trk"))):
        print("Loading {:.<20}".format(os.path.basename(trk_path)), end="\r")
        trk_file = nib.streamlines.load(trk_path)
        bundles.append(trk_file.tractogram)
        if i == 0:
            header = trk_file.header

    n_fibers = sum([len(b.streamlines) for b in bundles])
    n_bundles = len(bundles)

    print("Loaded {} fibers from {} bundles.".format(n_fibers, n_bundles))

    merged_bundles = bundles[0].copy()
    for b in bundles[1:]:
        merged_bundles.extend(b)

    if keep < 1:
        if weighted:
            p = np.zeros(n_fibers)
            offset=0
            for b in bundles:
                l = len(b.streamlines)
                p[offset:offset+l] = 1 / (l * n_bundles)
                offset += l
        else:
            p = np.ones(n_fibers) / n_fibers

        keep_n = int(keep * n_fibers)
        print("Subsampling {} fibers".format(keep_n))

        np.random.seed(42)
        subsample = np.random.choice(
            merged_bundles.streamlines,
            size=keep_n,
            replace=False,
            p=p)

        tractogram = Tractogram(
                streamlines=subsample,
                affine_to_rasmm=np.eye(4)
            )
    else:
        tractogram = merged_bundles

    if out_dir is None:
        out_dir = os.path.dirname(trk_dir)
        out_dir = os.path.join(out_dir, "merged_tracts")

    os.makedirs(out_dir, exist_ok=True)

    if weighted:
        save_path = os.path.join(out_dir,
            "merged_W{:03d}.trk".format(int(100*args.keep)))
    else:
        save_path = os.path.join(out_dir,
            "merged_{:03d}.trk".format(int(100*args.keep)))

    print("Saving {}".format(save_path))

    TrkFile(tractogram, header).save(save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Merge trk files.\n"
        "Merge several bundle trks with optional weighted subsampling.\n\n"
        "WARNING: Assumes that each trk file has the same affine.\n\n"
        "HCP whole brain ~ 1.700.000 fibers from 72 bundles\n"
        "keep=0.05 ~ 90k fibers ~ 3.5 Mio segments"
        "n_segments ~ 40 x n_fibers\n")

    parser.add_argument("trk_dir", help="Directory containing trk files.")

    parser.add_argument("--keep", default=1.0, type=float, 
        help="Fraction of fibers to keep during subsampling.")

    parser.add_argument("--weighted", action="store_true",
        help="Perform subsampling weighted by bundle size.")

    parser.add_argument("--out_dir",
        help="Directory for saving the merged .trk file.",
        type=str, default=None)

    args = parser.parse_args()

    assert args.keep >= 0.01

    merge_trks(args.trk_dir, args.keep, args.weighted, args.out_dir)