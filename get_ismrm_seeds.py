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
from dipy.io.utils import get_reference_info

def get_ismrm_seeds(data_dir, source, keep, weighted, threshold):

    trk_dir = os.path.join(data_dir, "bundles")

    if source in ["wm", "trk"]:
        anat_path = os.path.join(data_dir, "masks", "wm.nii.gz")
        resized_path = os.path.join(data_dir, "masks", "wm_125.nii.gz")
    elif source == "brain":
        anat_path = os.path.join("subjects", "ismrm_gt", "dwi_brain_mask.nii.gz")
        resized_path = os.path.join("subjects", "ismrm_gt",
            "dwi_brain_mask_125.nii.gz")  

    sp.call(["mrresize", "-voxel", "1.25", anat_path, resized_path])

    if source == "trk":

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

        if keep < 1:
            if weighted:
                p = np.zeros(n_seeds)
                offset=0
                for b in seed_bundles:
                    l = len(b)
                    p[offset:offset+l] = 1 / (l * n_bundles)
                    offset += l
            else:
                p = np.ones(n_seeds) / n_seeds

    elif source in ["brain","wm"]:

        weighted = False

        wm_file = nib.load(resized_path)
        wm_img = wm_file.get_fdata()

        seeds = np.argwhere(wm_img > threshold)
        seeds = np.hstack([seeds, np.ones([len(seeds), 1])])

        seeds = (wm_file.affine.dot(seeds.T).T)[:, :3].reshape(-1, 1, 3)

        n_seeds = len(seeds)

        if keep < 1:
            p = np.ones(n_seeds) / n_seeds

        header = TrkFile.create_empty_header()

        header["voxel_to_rasmm"] = wm_file.affine
        header["dimensions"] = wm_file.header["dim"][1:4]
        header["voxel_sizes"] = wm_file.header["pixdim"][1:4]
        header["voxel_order"] = get_reference_info(wm_file)[3]

    if keep < 1:
        keep_n = int(keep * n_seeds)
        print("Subsampling from {} seeds to {} seeds".format(n_seeds, keep_n))
        np.random.seed(42)
        keep_idx = np.random.choice(
            len(seeds),
            size=keep_n,
            replace=False,
            p=p)
        seeds = seeds[keep_idx]

    tractogram = Tractogram(
            streamlines=ArraySequence(seeds),
            affine_to_rasmm=np.eye(4)
        )

    save_dir=os.path.join(data_dir, "seeds")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "seeds_from_{}_{}{:03d}.trk")
    save_path = save_path.format(
        source,
        "W" if weighted else "",
        int(100*keep)
    )

    print("Saving {}".format(save_path))
    TrkFile(tractogram, header).save(save_path)

    os.remove(resized_path)
    for file in glob.glob(os.path.join(trk_dir, "*.trk")):
        os.remove(file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Convert fiber endpoints to seeds.")

    parser.add_argument("--data_dir", help="Path to scoring_data directory.",
        default="scoring/scoring_data")

    parser.add_argument("--source", default="wm", type=str,
        choices=["wm", "trk", "brain"], 
        help="Source for seeds: White Matter (wm) or Tracts (trk).")

    parser.add_argument("--keep", default=1.0, type=float, 
        help="Fraction of seeds to keep during subsampling.")

    parser.add_argument("--weighted", action="store_true",
        help="If provided, subsample seeds weighted by fiber bundle size. "
        "Only applicable if source == trk.")

    parser.add_argument("--thresh", default=0.1, type=float,
        help="Only applicable if source == wm. Threshold for White Matter "
        "Mask.")

    args = parser.parse_args()

    assert args.keep >= 0.01

    get_ismrm_seeds(
        args.data_dir,
        args.source,
        args.keep,
        args.weighted,
        args.thresh
    )