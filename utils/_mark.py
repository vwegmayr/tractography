import os
import argparse

import nibabel as nib
import numpy as np

from time import time

from tensorflow.keras.models import load_model

from nibabel.streamlines.trk import TrkFile
from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.streamlines.tractogram import (Tractogram, PerArraySequenceDict,
    PerArrayDict)

from models import MODELS
from utils.training import setup_env, maybe_get_a_gpu, timestamp
from utils.prediction import get_blocksize
from utils.config import load

from generate_samples import interpolate

import configs

@setup_env
def mark(model_path, fiber_path, dwi_path, gpu_queue=None):

    gpu_idx = maybe_get_a_gpu() if gpu_queue is None else gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

    dwi_img = nib.load(dwi_path)
    dwi_img = nib.funcs.as_closest_canonical(dwi_img)
    dwi_aff = dwi_img.affine
    dwi_affi = np.linalg.inv(dwi_aff)
    dwi = dwi_img.get_data()

    def xyz2ijk(coords, snap=False):

        ijk = (coords.T).copy()

        ijk = np.vstack([ijk, np.ones([1, ijk.shape[1]])])

        dwi_affi.dot(ijk, out=ijk)

        if snap:
            return (np.round(ijk, out=ijk).astype(int, copy=False).T)[:,:4]
        else:
            return (ijk.T)[:,:4]

    # ==========================================================================

    trk_file = nib.streamlines.load(fiber_path)
    tractogram = trk_file.tractogram
    streamlines = tractogram.streamlines
    tangents = tractogram.data_per_point["t"]


    n_fibers = len(tractogram)
    fiber_lengths = np.array([len(t.streamline) for t in tractogram])
    n_pts = fiber_lengths.sum()

    # ==========================================================================

    model_name = model_path.split("/")[1]

    if hasattr(MODELS[model_name], "custom_objects"):
        model = load_model(model_path,
                           custom_objects=MODELS[model_name].custom_objects,
                           compile=False)
    else:
        model = load_model(model_path, compile=False)

    block_size = get_blocksize(model, dwi.shape[-1])

    d = np.zeros([n_fibers, dwi.shape[-1] * block_size**3])
    dnorm = np.zeros([n_fibers, 1])

    kappa = [np.zeros([l, 1]) for l in fiber_lengths]
    log1p_kappa = [np.zeros([l, 1]) for l in fiber_lengths]
    log_prob = [np.zeros([l, 1]) for l in fiber_lengths]
    log_prob_map = [np.zeros([l, 1]) for l in fiber_lengths]

    max_step = fiber_lengths.max()
    step=0
    while step < max_step:
        t0 = time()

        left_idx  = np.where(step < fiber_lengths)[0]
        n_left = len(left_idx)

        # Get coords of latest segment for each ongoing fiber
        xyz = np.array([streamlines[i][step] for i in left_idx])
        ijk = xyz2ijk(xyz, snap=True)

        for ii, idx in enumerate(ijk):
            d[ii] = dwi[
                    idx[0]-(block_size // 2): idx[0]+(block_size // 2)+1,
                    idx[1]-(block_size // 2): idx[1]+(block_size // 2)+1,
                    idx[2]-(block_size // 2): idx[2]+(block_size // 2)+1,
                    :].flatten()  # returns copy
            dnorm[ii] = np.linalg.norm(d[ii])
            d[ii] /= dnorm[ii]

        if step == 0:
            vin = - np.array([tangents[i][step+1] for i in left_idx])
            vout = - np.array([tangents[i][step] for i in left_idx])
        else:
            vin = np.array([tangents[i][step-1] for i in left_idx])
            vout = np.array([tangents[i][step] for i in left_idx])

        inputs = np.hstack([vin, d[:n_left], dnorm[:n_left]])

        chunk = 2**15  # 32768
        n_chunks = np.ceil(n_left / chunk).astype(int)
        for c in range(n_chunks):

            fvm_pred, kappa_pred = model(inputs[c * chunk : (c + 1) * chunk])

            log1p_kappa_pred = np.log1p(kappa_pred)

            log_prob_pred = fvm_pred.log_prob(vout)

            log_prob_map_pred = fvm_pred._log_normalization() + kappa_pred

            for ii, i in enumerate(left_idx[c * chunk : (c + 1) * chunk]):

                kappa[i][step] = kappa_pred[ii]
                log1p_kappa[i][step] = log1p_kappa_pred[ii]
                log_prob[i][step] = log_prob_pred[ii]
                log_prob_map[i][step] = log_prob_map_pred[ii]

        print("Step {:3d}/{:3d} @ {:6.0f} steps/sec".format(
            step, max_step, n_left / (time() - t0)), end="\r")

        step += 1

    log_prob_ratio = [
        np.ones_like(log_prob[i]) * (log_prob[i].sum() / log_prob_map[i].sum())
        for i in range(n_fibers)
    ]

    data_per_point = PerArraySequenceDict(
        n_rows=n_pts,
        kappa=kappa,
        log1p_kappa=log1p_kappa,
        log_prob=log_prob,
        log_prob_map=log_prob_map,
        log_prob_ratio=log_prob_ratio
    )
    tractogram = Tractogram(
        streamlines=tractogram.streamlines,
        data_per_point=data_per_point,
        affine_to_rasmm=np.eye(4)
    )
    out_dir = os.path.join(
        os.path.dirname(dwi_path), "marked_fibers", timestamp()
    )
    os.makedirs(out_dir, exist_ok=True)

    marked_path = os.path.join(out_dir, "marked.trk")
    TrkFile(tractogram, trk_file.header).save(marked_path)

    config = dict(
        out_dir=out_dir,
        model_path=model_path,
        fiber_path=fiber_path,
        dwi_path=dwi_path
    )
    configs.save(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Mark fiber uncertainty.")

    parser.add_argument("model_path", type=str)

    parser.add_argument("fiber_path", type=str)

    parser.add_argument("dwi_path", type=str)

    args = parser.parse_args()

    mark(args.model_path, args.fiber_path, args.dwi_path)