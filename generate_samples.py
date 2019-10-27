import os
import random
import datetime

import nibabel as nib
import numpy as np
import yaml
import argparse

from scipy.interpolate import RegularGridInterpolator

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)


def interpolate(idx, dwi, block_size):

    IDX = np.round(idx).astype(int)

    values = np.zeros([3, 3, 3,
                       block_size, block_size, block_size,
                       dwi.shape[-1]])

    for x in range(3):
        for y in range(3):
            for z in range(3):
                values[x, y, z,:] = dwi[
                    IDX[0] + x - 2 * (block_size // 2) : IDX[0] + x + 1,
                    IDX[1] + y - 2 * (block_size // 2) : IDX[1] + y + 1,
                    IDX[2] + z - 2 * (block_size // 2) : IDX[2] + z + 1,
                    :]

    fn = RegularGridInterpolator(
        ([-1,0,1],[-1,0,1],[-1,0,1]), values)
    
    return (fn([idx[0]-IDX[0], idx[1]-IDX[1], idx[2]-IDX[2]])[0]).flatten()


def generate_conditional_samples(dwi,
                                 tracts,
                                 dwi_xyz2ijk,
                                 block_size,
                                 n_samples):

    fiber_lengths = [len(f) - 1 for f in tracts]
    n_samples = min(2*np.sum(fiber_lengths), n_samples)
    #===========================================================================
    inputs = np.zeros([n_samples, 3 + 1 + dwi.shape[-1] * block_size**3],
        dtype="float32")
    outgoing = np.zeros([n_samples, 3], dtype="float32")
    isterminal = np.zeros(n_samples, dtype="float32")
    done=False
    n = 0
    for tract in tracts:
        last_pt = len(tract.streamline) - 1
        for i, pt in enumerate(tract.streamline):
            #-------------------------------------------------------------------
            idx = dwi_xyz2ijk(pt)
            d = interpolate(idx, dwi, block_size)
            dnorm = np.linalg.norm(d)
            d /= dnorm
            #-------------------------------------------------------------------
            vout = tract.data_for_points["t"][i]

            if i == 0:
                vout *= -1
                vin = -tract.data_for_points["t"][i+1]
            else:
                vin = tract.data_for_points["t"][i-1]

            inputs[n] = np.hstack([vin, d, dnorm])
            outgoing[n] = vout
            if i in [0, last_pt]:
                isterminal[n] = 1
            n += 1

            if i not in [0, last_pt]:
                inputs[n] = np.hstack([-vin, d, dnorm])
                outgoing[n] = -vout
                n += 1
            #-------------------------------------------------------------------
            if n == n_samples:
                done = True
                break

        print("Finished {:3.0f}%".format(100*n/n_samples), end="\r")

        if done:
            return (n_samples,
                {"inputs": inputs, "isterminal": isterminal,
                 "outgoing": outgoing})


def generate_prior_samples(dwi,
                           tracts,
                           dwi_xyz2ijk,
                           block_size,
                           n_samples):

    n_samples = min(2*len(tracts), n_samples)
    #===========================================================================
    inputs = np.zeros([n_samples, 1 + dwi.shape[-1] * block_size**3],
        dtype="float32")
    outgoing = np.zeros([n_samples, 3], dtype="float32")
    done=False
    n=0
    for tract in tracts:
        for i, pt in enumerate(tract.streamline[[0, -1]]):
            #-------------------------------------------------------------------
            idx = dwi_xyz2ijk(pt)
            d = interpolate(idx, dwi, block_size)
            dnorm = np.linalg.norm(d)
            d /= dnorm
            #-------------------------------------------------------------------
            vout = tract.data_for_points["t"][i]
            if i == 1:
                vout *= -1
            inputs[n] = np.hstack([d, dnorm])
            outgoing[n] = vout
            n += 1
            #-------------------------------------------------------------------
            if n == (n_samples - 1):
                done = True
                break
            #-------------------------------------------------------------------
        print("Finished {:3.0f}%".format(100*n/n_samples), end="\r")

        if done:
            return n_samples, {"inputs": inputs, "outgoing": outgoing}


def generate_samples(dwi_path,
                     trk_path,
                     model,
                     block_size,
                     n_samples,
                     out_dir):
    """"""
    assert n_samples % 2 == 0

    trk_file = nib.streamlines.load(trk_path)
    assert trk_file.tractogram.data_per_point is not None
    assert "t" in trk_file.tractogram.data_per_point
    #===========================================================================
    dwi_img = nib.load(dwi_path)
    dwi_img = nib.funcs.as_closest_canonical(dwi_img)
    dwi_aff = dwi_img.affine
    dwi_affi = np.linalg.inv(dwi_aff)
    dwi_xyz2ijk = lambda r: dwi_affi.dot([r[0], r[1], r[2], 1])[:3]
    dwi = dwi_img.get_data()

    tracts = trk_file.tractogram # fiber coordinates in rasmm
    #===========================================================================
    if model == "conditional":
        n_samples, samples = generate_conditional_samples(dwi, tracts,
            dwi_xyz2ijk, block_size, n_samples)
    elif model == "prior":
        n_samples, samples = generate_prior_samples(dwi, tracts, dwi_xyz2ijk,
            block_size, n_samples)
    #===========================================================================
    np.random.seed(42)
    perm = np.random.permutation(n_samples)
    for k, v in samples.items():
        assert not np.isnan(v).any()
        assert not np.isinf(v).any()
        samples[k] = v[perm]
    #===========================================================================
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(dwi_path), "samples")
    out_dir = os.path.join(out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    sample_path = os.path.join(out_dir, "samples.npz")
    print("\nSaving {}".format(sample_path))
    np.savez_compressed(
        sample_path,
        input_shape=samples["inputs"].shape[1:],
        n_samples=n_samples,
        **samples)
    
    config_path = os.path.join(out_dir, "config.yml")
    config=dict(
        n_samples=int(n_samples),
        dwi_path=dwi_path,
        trk_path=trk_path,
        model=model,
        block_size=int(block_size),
    )
    print("Saving {}".format(config_path))
    with open(config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
            
    return samples


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Generate sample npz from DWI and TRK data.")

    parser.add_argument("dwi_path", help="Path to DWI file")

    parser.add_argument("trk_path", help="Path to TRK file")

    parser.add_argument("--model", default="conditional",
        choices=["conditional", "prior"],
        help="Which model to generate samples for.")

    parser.add_argument("--block_size", help="Size of cubic neighborhood.",
        default=3, choices=[1,3,5,7], type=int)

    parser.add_argument("--n_samples", default=2**30, type=int,
        help="Maximum number of samples to keep.")

    parser.add_argument("--out_dir", default=None, 
        help="Sample directory, by default creates directory next to dwi_path.")

    args = parser.parse_args()

    generate_samples(
        args.dwi_path,
        args.trk_path,
        args.model,
        args.block_size,
        args.n_samples,
        args.out_dir)
