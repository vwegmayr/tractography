import os
os.environ['PYTHONHASHSEED'] = '0'
import argparse
import nibabel as nib
import nipy as ni
import numpy as np


parser = argparse.ArgumentParser(description="Convert WM mask to seeds.")

parser.add_argument("wm_path", help="Path to WM mask.")

parser.add_argument("--samples", default=1, type=int, 
    help="Number of random seeds per WM voxel.")

args = parser.parse_args()

np.random.seed(42)

wm_img = ni.load_image(args.wm_path)
ijk2xyz = lambda r: wm_img.coordmap(r)

wm = wm_img.get_data()

ijk = np.argwhere(wm == 1)
seeds = ijk2xyz(ijk)
seeds = np.hstack([seeds, np.ones((len(seeds), 1))]) # Additional dimension for 4x4 affine

save_path = os.path.join(os.path.dirname(args.wm_path), "wm_seeds.npy")

print("Saving {}".format(save_path))
np.save(save_path, seeds)