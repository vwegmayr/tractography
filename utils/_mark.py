import os
import argparse

import nibabel as nib
import numpy as np

from time import time

from tensorflow.keras.models import load_model

from nibabel.streamlines.trk import TrkFile
from nibabel.streamlines.tractogram import (Tractogram, PerArraySequenceDict)

from resample_trk import maybe_add_tangent
from models import MODELS
from utils.training import setup_env, maybe_get_a_gpu, timestamp
from utils.prediction import get_blocksize
from utils.config import load

import configs


@setup_env
def mark(config, gpu_queue=None):

    gpu_idx = -1
    try:
        gpu_idx = maybe_get_a_gpu() if gpu_queue is None else gpu_queue.get()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    except Exception as e:
        print(str(e))
    print("Loading DWI data ...")

    dwi_img = nib.load(config["dwi_path"])
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

    print("Loading fibers ...")

    trk_file = nib.streamlines.load(config["trk_path"])
    tractogram = trk_file.tractogram

    if "t" in tractogram.data_per_point:
        print("Fibers are already resampled")
        tangents = tractogram.data_per_point["t"]
    else:
        print("Fibers are not resampled. Resampling now ...")
        tractogram = maybe_add_tangent(config["trk_path"],
                                       min_length=30,
                                       max_length=200)
        tangents = tractogram.data_per_point["t"]

    n_fibers = len(tractogram)
    fiber_lengths = np.array([len(t.streamline) for t in tractogram])
    max_length = fiber_lengths.max()
    n_pts = fiber_lengths.sum()

    # ==========================================================================

    print("Loading model ...")
    model_name = config['model_name']

    if hasattr(MODELS[model_name], "custom_objects"):
        model = load_model(config["model_path"],
                           custom_objects=MODELS[model_name].custom_objects,
                           compile=False)
    else:
        model = load_model(config["model_path"], compile=False)

    block_size = get_blocksize(model, dwi.shape[-1])

    d = np.zeros([n_fibers, dwi.shape[-1] * block_size**3 + 1])

    inputs = np.zeros([
        n_fibers,
        max_length,
        3]
    )

    print("Writing to input array ...")

    for i, fiber_t in enumerate(tangents):
        inputs[i, :fiber_lengths[i], :] = fiber_t

    outputs = np.zeros([
        n_fibers,
        max_length,
        4]
    )

    print("Starting iteration ...")

    step=0
    while step < max_length:
        t0 = time()

        xyz = inputs[:, step, :]
        ijk = xyz2ijk(xyz, snap=True)

        for ii, idx in enumerate(ijk):
            try:
                d[ii, :-1] = dwi[
                        idx[0]-(block_size // 2): idx[0]+(block_size // 2)+1,
                        idx[1]-(block_size // 2): idx[1]+(block_size // 2)+1,
                        idx[2]-(block_size // 2): idx[2]+(block_size // 2)+1,
                        :].flatten()  # returns copy
            except (IndexError, ValueError):
                pass

        d[:, -1] = np.linalg.norm(d[:, :-1], axis=1) + 10**-2

        d[:, :-1] /= d[:, -1].reshape(-1, 1)

        if step == 0:
            vin = - inputs[:, step+1, :]
            vout = - inputs[:, step, :]
        else:
            vin = inputs[:, step-1, :]
            vout = inputs[:, step, :]

        model_inputs = np.hstack([vin, d])
        chunk = 2**15  # 32768
        n_chunks = np.ceil(n_fibers / chunk).astype(int)
        for c in range(n_chunks):

            fvm_pred, kappa_pred = model(model_inputs[c * chunk : (c + 1) * chunk])

            log1p_kappa_pred = np.log1p(kappa_pred)

            log_prob_pred = fvm_pred.log_prob(vout[c * chunk : (c + 1) * chunk])

            log_prob_map_pred = fvm_pred._log_normalization() + kappa_pred

            outputs[c * chunk : (c + 1) * chunk, step, 0] = kappa_pred
            outputs[c * chunk : (c + 1) * chunk, step, 1] = log1p_kappa_pred
            outputs[c * chunk : (c + 1) * chunk, step, 2] = log_prob_pred
            outputs[c * chunk : (c + 1) * chunk, step, 3] = log_prob_map_pred

        print("Step {:3d}/{:3d}, ETA: {:4.0f} min".format(
            step, max_length, (max_length-step)*(time()-t0)/60 ), end="\r")

        step += 1

    if gpu_queue is not None:
        gpu_queue.put(gpu_idx)

    kappa = [outputs[i, :fiber_lengths[i], 0].reshape(-1, 1)
        for i in range(n_fibers)]
    log1p_kappa = [outputs[i, :fiber_lengths[i], 1].reshape(-1, 1)
        for i in range(n_fibers)]
    log_prob = [outputs[i, :fiber_lengths[i], 2].reshape(-1, 1)
        for i in range(n_fibers)]
    log_prob_map = [outputs[i, :fiber_lengths[i], 3].reshape(-1, 1) 
        for i in range(n_fibers)]

    log_prob_sum = [
        np.ones_like(log_prob[i]) * (log_prob[i].sum() / log_prob_map[i].sum())
        for i in range(n_fibers)
    ]
    log_prob_ratio = [
        np.ones_like(log_prob[i]) * (log_prob[i] - log_prob_map[i]).mean()
        for i in range(n_fibers)
    ]

    other_data={}
    for key in list(trk_file.tractogram.data_per_point.keys()):
        if key not in ["kappa", "log1p_kappa", "log_prob", "log_prob_map",
                       "log_prob_sum", "log_prob_ratio"]:
            other_data[key] = trk_file.tractogram.data_per_point[key]

    data_per_point = PerArraySequenceDict(
        n_rows=n_pts,
        kappa=kappa,
        log_prob=log_prob,
        log_prob_sum=log_prob_sum,
        log_prob_ratio=log_prob_ratio,
        **other_data
    )
    tractogram = Tractogram(
        streamlines=tractogram.streamlines,
        data_per_point=data_per_point,
        affine_to_rasmm=np.eye(4)
    )
    out_dir = os.path.join(
        os.path.dirname(config["dwi_path"]), "marked_fibers", timestamp()
    )
    os.makedirs(out_dir, exist_ok=True)

    marked_path = os.path.join(out_dir, "marked.trk")
    TrkFile(tractogram, trk_file.header).save(marked_path)

    config["out_dir"] = out_dir

    configs.save(config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Mark fiber statistics.")

    parser.add_argument("config_path", type=str)

    args = parser.parse_args()

    config = load(args.config_path)

    mark(config)
