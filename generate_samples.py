import os
os.environ['PYTHONHASHSEED'] = '0'
import random as rn
rn.seed(12345)
from numpy.random import seed
seed(42)

import nibabel as nib
import nipy as ni
import numpy as np
import datetime
import shutil
import yaml
import csv
import json
import argparse

from hashlib import md5
from scipy.interpolate import RegularGridInterpolator

def generate_samples(
    dwi_path,
    trk_path,
    model,
    block_size,
    no_reverse,
    keep_n,
    out_dir):
    """"""
    hasher = md5()
    hasher.update(str(dwi_path).encode())
    hasher.update(str(trk_path).encode())
    hasher.update(str(model).encode())
    hasher.update(str(block_size).encode())
    hasher.update(str(no_reverse).encode())
    hasher.update(str(keep_n).encode())

    if not no_reverse and keep_n % 2 != 0:
        raise ValueError("keep_n can not be an odd number for "
            "no_reverse == False.")
    
    trk_file = nib.streamlines.load(trk_path)
    assert trk_file.tractogram.data_per_point is not None
    assert "t" in trk_file.tractogram.data_per_point
    
    #=================================================
    
    dwi_img = nib.load(dwi_path)
    dwi_img = nib.funcs.as_closest_canonical(dwi_img)
    dwi_aff = dwi_img.affine
    dwi_affi = np.linalg.inv(dwi_aff)
    dwi_xyz2ijk = lambda r: dwi_affi.dot([r[0], r[1], r[2], 1])[:3]
    dwi = dwi_img.get_data()

    tracts = trk_file.tractogram # fiber coordinates in rasmm

    n_fibers = len(tracts)
    fiber_lengths = [len(f) for f in tracts]
    n_samples = np.sum(fiber_lengths) - 2 * n_fibers
    if not no_reverse:
        n_samples *= 2 
    n_samples = min(n_samples, keep_n)
    
    np.random.seed(42)
    perm = np.random.permutation(len(tracts))
    tracts = tracts[perm]
    
    #=================================================
    
    inputs = []
    outputs = []
    done=False
    for fi, f in enumerate(tracts):  
        for i, r in enumerate(f.streamline[1:-1]): # Exclude end points for conditional model
            try:
                idx = dwi_xyz2ijk(r) # anchor idx
                IDX = np.round(idx).astype(int)
                
                values = np.zeros([3, 3, 3,
                                   block_size, block_size, block_size,
                                   dwi.shape[-1]])
                
                for x in range(block_size):
                    for y in range(block_size):
                        for z in range(block_size):
                            values[x, y, z,:] = dwi[
                                IDX[0] + x - 2 * (block_size // 2) : IDX[0] + x + 1,
                                IDX[1] + y - 2 * (block_size // 2) : IDX[1] + y + 1,
                                IDX[2] + z - 2 * (block_size // 2) : IDX[2] + z + 1,
                                :]
                fn = RegularGridInterpolator(
                    ([-1,0,1],[-1,0,1],[-1,0,1]), values)
                
                d = fn([idx[0]-IDX[0], idx[1]-IDX[1], idx[2]-IDX[2]])[0]
                d = d.flatten() # to get back the spatial order: reshape(bs, bs, bs, dwi.shape[-1])
                
            except IndexError:
                n_samples -= (1 if no_reverse else 2)
                print(("Index error at r={}, idx={}, fiber_idx={}\nMaybe wrong "
                    "reference frame, or resampling failed.").format(
                    r, idx, perm[fi]))
                continue
                
            vout = f.data_for_points["t"][i+1].astype("float32")
            vin = f.data_for_points["t"][i].astype("float32")

            outputs.append(vout)
            inputs.append(np.hstack([vin, d]).astype("float32"))

            if not no_reverse:
                inputs.append(np.hstack([-vout, d]).astype("float32"))
                outputs.append(-vin)

            if len(inputs) == n_samples:
                done = True
                break

        print("Finished {:3.0f}%".format(100*len(inputs)/n_samples), end="\r")

        if done:
            break

    assert n_samples == len(inputs)
    assert n_samples == len(outputs)
    assert inputs[0].shape == (3 + dwi_img.shape[-1] * block_size**3, )
    assert outputs[0].shape == (3, )

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(dwi_path), "samples")
    out_dir = os.path.join(out_dir, hasher.hexdigest())
    os.makedirs(out_dir, exist_ok=True)

    sample_path = os.path.join(out_dir, "samples.npz")
    print("\nSaving {}".format(sample_path))
    np.savez_compressed(sample_path, inputs=inputs, outputs=outputs)
    
    config_path = os.path.join(out_dir, "config.yml")
    config=dict(
        n_samples=int(n_samples),
        dwi_path=dwi_path,
        trk_path=trk_path,
        model=model,
        block_size=int(block_size),
        no_reverse=str(no_reverse)
    )
    print("Saving {}".format(config_path))
    with open(config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
            
    return inputs, outputs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Generate sample npz from DWI and TRK data.")

    parser.add_argument("dwi_path", help="Path to DWI file")

    parser.add_argument("trk_path", help="Path to TRK file")

    parser.add_argument("--model", default="conditional",
        choices=["conditional"], help="Which model to generate samples for.")

    parser.add_argument("--block_size", help="Size of cubic neighborhood.",
        default=3, choices=[1,3,5,7], type=int)

    parser.add_argument("--no_reverse", action="store_false",
        help="Do not include direction-reversed samples.")

    parser.add_argument("--keep_n", default=2**30, type=int,
        help="Maximum number of samples to keep.")

    parser.add_argument("--out_dir", default=None, 
        help="Sample directory, by default creates directory next to dwi_path.")

    args = parser.parse_args()

    generate_samples(
        args.dwi_path,
        args.trk_path,
        args.model,
        args.block_size,
        args.no_reverse,
        args.keep_n,
        args.out_dir)
