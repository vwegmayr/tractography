import argparse
import glob
import os

import numpy as np
import nibabel as nib

from nibabel.streamlines.tractogram import Tractogram

parser = argparse.ArgumentParser(description="Merge trk files.\n"
    "Merge several bundle trks with optional weighted subsampling.\n"
    "WARNING: Does not perform any checks, e.g. whether fibers are aligned.")

parser.add_argument("trk_dir", help="Directory containing trk files.")

parser.add_argument("--keep", default=0.5, type=float, 
    help="Fraction of fibers to keep during subsampling.")

parser.add_argument("--weighted", action="store_true",
    help="Perform subsampling weighted by bundle size.")

args = parser.parse_args()

bundles = []
for trk_file in glob.glob(os.path.join(args.trk_dir, "*")):
    print("Loading {:15} ...".format(os.path.basename(trk_file)), end="\r")
    bundles.append(nib.streamlines.load(trk_file).tractogram)

n_fibers = sum([len(b.streamlines) for b in bundles])
n_bundles = len(bundles)

print("Loaded {} fibers from {} bundles.".format(n_fibers, n_bundles))

if args.weighted:
    p = np.zeros(n_fibers)
    offset=0
    for b in bundles:
        l = len(b.streamlines)
        p[offset:offset+l] = 1 / (l * n_bundles)
        offset += l
else:
    p = np.ones(n_fibers) / n_fibers

merged_bundles = bundles[0].copy()
for b in bundles[1:]:
    merged_bundles.extend(b)

keep_n = int(args.keep * n_fibers)
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

if args.weighted:
    save_path = os.path.join(trk_dir,
        "merged_w{}.trk".format(int(100*args.keep)))
else:
    save_path = os.path.join(trk_dir,
        "merged_{}.trk".format(int(100*args.keep)))

print("Saving {}".format(save_path))

nib.streamlines.save(tractogram, save_path)