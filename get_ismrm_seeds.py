import os
os.environ['PYTHONHASHSEED'] = '0'
import glob
import argparse
import nibabel as nib
import numpy as np
import subprocess as sp
from hashlib import md5
from nibabel.streamlines import ArraySequence, Tractogram
from nibabel.streamlines.trk import TrkFile


parser = argparse.ArgumentParser(description="Convert fiber endpoints to seeds.")

parser.add_argument("--data_dir", help="Path to scoring_data directory.",
    default="scoring/scoring_data")

parser.add_argument("--keep", default=1.0, type=float, 
    help="Fraction of seeds to keep during subsampling.")

parser.add_argument("--weighted", action="store_true",
    help="If provided, subsample seeds weighted by bundle size.")

args = parser.parse_args()


assert args.keep >= 0.01

trk_dir = os.path.join(args.data_dir, "bundles")

anat_path = os.path.join(args.data_dir, "masks", "wm.nii.gz")
resized_path = os.path.join(args.data_dir, "masks", "wm_125.nii.gz")
sp.call(["mrresize", "-voxel", "1.25", anat_path, resized_path])

print("Running Tractconverter...")
sp.call([
    "python",
    "tractconverter/scripts/WalkingTractConverter.py",
    "-i", trk_dir,
    "-a", resized_path,
    "-vtk2trk"])

print("Loading seed bundles...")
seed_bundles = []
for i, trk_path in enumerate(glob.glob(os.path.join(trk_dir, "*.trk"))):
    trk_file = nib.streamlines.load(trk_path)
    endpoints = []
    for fiber in trk_file.tractogram.streamlines:
        endpoints.append(fiber[0])
        endpoints.append(fiber[-1])
    seed_bundles.append(endpoints)
    if i == 0:
        header = trk_file.header

n_seeds = sum([len(b) for b in seed_bundles])
n_bundles = len(seed_bundles)

print("Loaded {} seeds from {} bundles.".format(n_seeds, n_bundles))

seeds = np.array([[seed] for bundle in seed_bundles for seed in bundle])

if args.keep < 1:
    if args.weighted:
        p = np.zeros(n_seeds)
        offset=0
        for b in seed_bundles:
            l = len(b)
            p[offset:offset+l] = 1 / (l * n_bundles)
            offset += l
    else:
        p = np.ones(n_seeds) / n_seeds

    keep_n = int(args.keep * n_seeds)
    print("Subsampling from {} seeds to {} seeds".format(n_seeds, keep_n))

    np.random.seed(42)
    seeds = np.random.choice(
        seeds,
        size=keep_n,
        replace=False,
        p=p)

tractogram = Tractogram(
        streamlines=ArraySequence(seeds),
        affine_to_rasmm=np.eye(4)
    )

save_dir=os.path.join(args.data_dir, "seeds")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "seeds_from_fibers_{}{}.trk")
save_path = save_path.format("w" if args.weighted else "", int(100*args.keep))
print("Saving {}".format(save_path))
TrkFile(tractogram, header).save(save_path)


os.remove(resized_path)
for file in glob.glob(os.path.join(trk_dir, "*.trk")):
    os.remove(file)