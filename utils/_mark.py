import os
import argparse

import nibabel as nib
import numpy as np

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

    dwi_xyz2ijk = lambda r: dwi_affi.dot([r[0], r[1], r[2], 1])[:3]

    # ==========================================================================

    trk_file = nib.streamlines.load(fiber_path)

    tractogram = trk_file.tractogram

    tractogram = tractogram[:15]

    n_fibers = len(tractogram)
    n_pts = np.sum([len(t.streamline) for t in tractogram])

    # ==========================================================================

    model_name = model_path.split("/")[1]

    if hasattr(MODELS[model_name], "custom_objects"):
        model = load_model(model_path,
                           custom_objects=MODELS[model_name].custom_objects,
                           compile=False)
    else:
        model = load_model(model_path, compile=False)

    block_size = get_blocksize(model, dwi.shape[-1])

    kappa = ArraySequence()
    log1p_kappa = ArraySequence()
    log_prob = ArraySequence()
    logp_ratio = ArraySequence()

    for i, tract in enumerate(tractogram):

        k = np.zeros([len(tract.streamline), 1])
        log1pk = np.zeros([len(tract.streamline), 1])
        logp = np.zeros([len(tract.streamline), 1])
        logpm = np.zeros([len(tract.streamline), 1])

        for j, r in enumerate(tract.streamline):

            if j == 0:
                vin = -tract.data_for_points["t"][j+1]
                vout = -tract.data_for_points["t"][j].reshape(1, -1)
            else:
                vin = tract.data_for_points["t"][j-1]
                vout = tract.data_for_points["t"][j].reshape(1, -1)

            idx = dwi_xyz2ijk(r)
            d = interpolate(idx, dwi, block_size)
            dnorm = np.linalg.norm(d)
            d /= dnorm

            inputs = np.hstack([vin, d, dnorm]).reshape(1, -1)

            fvm, k_pred = model(inputs)

            k[j, 0] = k_pred[0]
            log1pk[j, 0] = np.log1p(k_pred[0])
            logp[j, 0] = fvm.log_prob(vout)[0]
            logpm[j, 0] = fvm._log_normalization()[0] + k_pred[0]

        kappa.append(k, cache_build=True)
        log1p_kappa.append(log1pk, cache_build=True)
        log_prob.append(logp, cache_build=True)
        logp_ratio.append(
            np.ones_like(logp) * logp.sum() / logpm.sum(),
            cache_build=True
        )

        print("{}".format(i), end="\r")

    kappa.finalize_append()
    log1p_kappa.finalize_append()
    log_prob.finalize_append()
    logp_ratio.finalize_append()

    data_per_point = PerArraySequenceDict(
        n_rows=n_pts,
        kappa=kappa,
        log1p_kappa=log1p_kappa,
        logp=log_prob,
        logp_ratio=logp_ratio
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