import os
import yaml
import gc
import argparse

import nibabel as nib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from nibabel.streamlines.trk import TrkFile
from nibabel.streamlines.array_sequence import ArraySequence, concatenate
from nibabel.streamlines.tractogram import Tractogram

from GPUtil import getFirstAvailable
from hashlib import md5
from sklearn.preprocessing import normalize
from time import time

os.environ['PYTHONHASHSEED'] = '0'
tf.compat.v1.set_random_seed(42)
np.random.seed(42)
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getFirstAvailable(
        order="load", maxLoad=10 ** -6, maxMemory=10 ** -1)[0])
except Exception as e:
    print(str(e))


class Prior(object):

    def __init__(self, prior_path):
        if ".nii" in prior_path:
            vec_img = nib.load(prior_path)
            self.vec = vec_img.get_data()
            affi = np.linalg.inv(vec_img.affine)
            self.xyz2ijk = lambda xyz: np.round(affi.dot(xyz.T).T).astype(int)
        elif ".h5" in prior_path:
            raise NotImplementedError # TODO: Implement prior model
            
    def __call__(self, xyz):
        if hasattr(self, "vec"):
            ijk = self.xyz2ijk(xyz)
            vecs = self.vec[ijk[:,0], ijk[:,1], ijk[:,2]]
            # Assuming that seeds have been duplicated for both directions!
            vecs[len(ijk)//2:, :] *= -1
            return normalize(vecs)
        elif hasattr(self, "model"):
            raise NotImplementedError # TODO: Implement prior model


class Terminator(object):

    def __init__(self, term_path, thresh):
        if ".nii" in term_path:
            scalar_img = nib.load(term_path)
            self.scalar = scalar_img.get_data()
            affi = np.linalg.inv(scalar_img.affine)
            self.xyz2ijk = lambda xyz: np.round(affi.dot(xyz.T).T).astype(int)
        elif ".h5" in term_path:
            raise NotImplementedError # TODO: Implement termination model
        self.threshold = thresh

    def __call__(self, xyz):
        if hasattr(self, "scalar"):
            ijk = self.xyz2ijk(xyz)
            return np.where(
                self.scalar[
                    ijk[:,0], ijk[:,1], ijk[:,2]] < self.threshold)[0]
        else:
            raise NotImplementedError


def run_inference(
    model_path,
    dwi_path,
    prior_path,
    seed_path,
    term_path,
    thresh,
    predict_fn,
    step_size,
    max_steps,
    out_dir
):
    """"""
    
    print("Loading DWI...")
    
    dwi_img = nib.load(dwi_path)

    dwi_img = nib.funcs.as_closest_canonical(dwi_img)
    dwi_aff = dwi_img.affine
    dwi_affi = np.linalg.inv(dwi_aff)
    dwi = dwi_img.get_data()
    
    print("Loading Model...")
    
    # TODO: put different losses into an outside module, and import them
    # with importlib according to a string passed by args.
    def negative_log_likelihood(observed_y, predicted_distribution):
        return -K.mean(predicted_distribution.log_prob(observed_y))
    
    model = load_model(model_path,
                       custom_objects={
                       "negative_log_likelihood": negative_log_likelihood,
                        "DistributionLambda": tfp.layers.DistributionLambda})
    
    # Define coordinate transforms
    
    input_shape = model.layers[0].get_output_at(0).get_shape().as_list()[-1]
    block_size = int(np.cbrt(input_shape / dwi.shape[-1]))
    
    def xyz2ijk(coords, snap=False, shift=False):
        ijk = dwi_affi.dot(coords.T).T
        if snap:
            ijk = np.round(ijk).astype(int)
        return ijk
    
    # Define Fiber Termination
    
    terminator = Terminator(term_path, thresh)
    
    print("Loading Seeds...")
    
    seed_file = nib.streamlines.load(seed_path)
    seeds = seed_file.tractogram.streamlines.data
    seeds = np.vstack([seeds, seeds])  # Duplicate seeds for both directions
    seeds = np.hstack([seeds, np.ones([len(seeds), 1])]) # add affine dimension
    assert seeds.shape[-1] == 4   # (x, y, z, 1)
    
    # Define Prior for First Fiber Direction
        
    prior = Prior(prior_path)
    
    print("Initialize Fibers...")
    
    xyz = seeds.reshape(-1, 1, 4) # (fiber, segment, coord)
    
    fiber_idx = np.hstack([np.arange(len(seeds)//2), np.arange(len(seeds)//2)])
    fibers = [[] for _ in range(len(seeds)//2)]
    
    print("Start Iteration...")
    
    for i in range(max_steps):
        t0 = time()
        
        ijk = xyz2ijk(xyz[:,-1,:], snap=True, shift=True) # Get coords of latest segement for each fiber 

        d = np.zeros([len(ijk), block_size, block_size, block_size, dwi.shape[-1]])

        for ii, idx in enumerate(ijk):
            d[ii] = dwi[idx[0] - (block_size // 2) : idx[0] + (block_size // 2) + 1,
                        idx[1] - (block_size // 2) : idx[1] + (block_size // 2) + 1,
                        idx[2] - (block_size // 2) : idx[2] + (block_size // 2) + 1,
                    :]
        d = d.reshape(-1, dwi.shape[-1] * block_size**3)
        
        if i == 0:
            vin = prior(xyz[:,0,:])
        else:
            vin = vout.copy()
        
        chunk_size = 2**15 # 32768
        n_chunks = np.ceil(len(vin) / chunk_size).astype(int)
        
        inputs = np.hstack([vin,d])
        vout = np.zeros([len(vin), 3])
        for chunk in range(n_chunks):
            input_chunk = inputs[chunk * chunk_size : (chunk + 1) * chunk_size]
            if predict_fn == "mean":
                v = model(input_chunk).mean().numpy()
                v = normalize(v) # Careful, the FvM mean is not a unit vector!
            else:
                v = model(input_chunk).sample().numpy() # Samples are unit length, though!
            vout[chunk * chunk_size : (chunk + 1) * chunk_size] = v
           
        rout = (xyz[:, -1, :3] + step_size * vout)
        rout = np.hstack([rout, np.ones((len(rout), 1))]).reshape(-1, 1, 4)
        
        xyz = np.concatenate([xyz, rout], axis=1)
        
        terminal_indices = terminator(xyz[:, -1, :]) # Check latest points for termination

        for idx in terminal_indices:
            gidx = fiber_idx[idx]
            # Other end not yet added
            if not fibers[gidx]:
                fibers[gidx].append(xyz[idx, :, :3])
            # Other end already added
            else:
                this_end = xyz[idx, :, :3]
                other_end = fibers[gidx][0]
                merged_fiber = np.vstack([
                    np.flip(this_end[1:], axis=0),
                    other_end]) # stitch ends together
                fibers[gidx] = [merged_fiber]
                
        xyz = np.delete(xyz, terminal_indices, axis=0)
        vout = np.delete(vout, terminal_indices, axis=0)
        fiber_idx = np.delete(fiber_idx, terminal_indices)
        
        print("Iter {:4d}/{}, finished {:5d}/{:5d} ({:3.0f}%) of all seeds with"
            " {:6.0f} steps/sec".format(
            (i+1), max_steps, len(seeds)-len(fiber_idx), len(seeds),
            100*(1-len(fiber_idx)/len(seeds)), len(vin) / (time() - t0)),
            end="\r"
        )
        
        if len(fiber_idx) == 0:
            break
            
        gc.collect()
    
    # Include unfinished fibers:
    
    for idx, gidx in enumerate(fiber_idx):
        if not fibers[gidx]:
            fibers[gidx].append(xyz[idx, :, :3])
        else:
            this_end = xyz[idx, :, :3]
            other_end = fibers[gidx][0]
            merged_fiber = np.vstack([np.flip(this_end[1:], axis=0), other_end])
            fibers[gidx] = [merged_fiber]
    
    # Save Result
    
    fibers = [f[0] for f in fibers]
    
    tractogram = Tractogram(
        streamlines=ArraySequence(fibers),
        affine_to_rasmm=np.eye(4)
    )
    
    if out_dir is None:
        out_dir = os.path.dirname(dwi_path)

    hasher = md5()
    hasher.update(model_path.encode())
    hasher.update(dwi_path.encode())
    hasher.update(prior_path.encode())
    hasher.update(seed_path.encode())
    hasher.update(str(term_path).encode())
    hasher.update(str(thresh).encode())
    hasher.update(predict_fn.encode())
    hasher.update(str(step_size).encode())
    hasher.update(str(max_steps).encode())

    out_dir = os.path.join(out_dir, "predicted_fibers", hasher.hexdigest())

    os.makedirs(out_dir, exist_ok=True)

    fiber_path = os.path.join(out_dir, "fibers.trk")
    print("\nSaving {}".format(fiber_path))
    TrkFile(tractogram, seed_file.header).save(fiber_path)

    config=dict(
        model_path=model_path,
        dwi_path=dwi_path,
        prior_path=prior_path,
        seed_path=seed_path,
        term_path=term_path,
        thresh=thresh,
        predict_fn=predict_fn,
        step_size=step_size,
        max_steps=max_steps
        )

    config_path = os.path.join(out_dir, "config.yml")
    print("Saving {}".format(config_path))
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False) 
    
    return tractogram


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Use a trained model to "
        "predict fibers on DWI data."
        )

    parser.add_argument("model_path", type=str,
        help="Path to stored keras model.")

    parser.add_argument("dwi_path", type=str,
        help="Path to DWI data.")

    parser.add_argument("prior_path", type=str,
        help="Path to prior file with either .nii or .h5 extension."
            "If .nii, assumes a WxLxHx3 volume containing the prior directions."
            "If .h5, assumes a trained model, which takes only DWI as input.")

    parser.add_argument("seed_path", type=str,
        help="Path to seed file (.trk).")

    parser.add_argument("term_path", type=str,
        help="Path to terminator file (.nii).")

    parser.add_argument("--thresh", type=float, default=0.1,
        help="Stopping threshold, used together with term_path, if provided.")

    parser.add_argument("--predict_fn", type=str, default="mean",
        choices=["mean", "sample"],
        help="Next-Step prediction mode, either along most-likely direction "
        "(mean), or along a randomly sampled direction (sample).")

    parser.add_argument("--step_size", type=float, default=0.25,
        help="Length of each step.")

    parser.add_argument("--max_steps", type=int, default=400,
        help="Maximum number of iterations.")

    parser.add_argument("--out_dir", type=str, default=None,
        help="Directory to save the predicted fibers. "
        "By default, it is created next to dwi_path.")

    args = parser.parse_args()

    run_inference(
        args.model_path,
        args.dwi_path,
        args.prior_path,
        args.seed_path,
        args.term_path,
        args.thresh,
        args.predict_fn,
        args.step_size,
        args.max_steps,
        args.out_dir
    )