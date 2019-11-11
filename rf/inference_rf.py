import os
import gc
import argparse
import datetime
import yaml

import nibabel as nib
import numpy as np
import pickle

from tensorflow.keras import backend as K

from nibabel.streamlines.trk import TrkFile
from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.streamlines.tractogram import Tractogram

from time import time
from dipy.io.gradients import read_bvals_bvecs

from utils.prediction import Prior, Terminator
from utils.training import setup_env, maybe_get_a_gpu
from utils._score import score_on_tm

import configs


@setup_env
def run_rf_inference(config=None, gpu_queue=None):
    """"""
    try:
        gpu_idx = maybe_get_a_gpu() if gpu_queue is None else gpu_queue.get()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    except Exception as e:
        print(str(e))

    print(
        "Loading DWI...")  ####################################################

    dwi_img = nib.load(config['dwi_path'])
    dwi_img = nib.funcs.as_closest_canonical(dwi_img)
    dwi_aff = dwi_img.affine
    dwi_affi = np.linalg.inv(dwi_aff)
    dwi = dwi_img.get_data()

    def xyz2ijk(coords, snap=False):
        ijk = (coords.T).copy()
        dwi_affi.dot(ijk, out=ijk)
        if snap:
            return np.round(ijk, out=ijk).astype(int, copy=False).T
        else:
            return ijk.T

    with open(os.path.join(config['model_dir'], 'model'), 'rb') as f:
        model = pickle.load(f)

    train_config_file = os.path.join(config['model_dir'], 'config.yml')
    bvec_path = configs.load(train_config_file, 'bvecs')
    _, bvecs = read_bvals_bvecs(None, bvec_path)

    terminator = Terminator(config['term_path'], config['thresh'])

    prior = Prior(config['prior_path'])

    print(
        "Initializing Fibers...")  ############################################

    seed_file = nib.streamlines.load(config['seed_path'])
    xyz = seed_file.tractogram.streamlines.data
    n_seeds = 2 * len(xyz)
    xyz = np.vstack([xyz, xyz])  # Duplicate seeds for both directions
    xyz = np.hstack([xyz, np.ones([n_seeds, 1])])  # add affine dimension
    xyz = xyz.reshape(-1, 1, 4)  # (fiber, segment, coord)

    fiber_idx = np.hstack([
        np.arange(n_seeds // 2, dtype="int32"),
        np.arange(n_seeds // 2, dtype="int32")
    ])
    fibers = [[] for _ in range(n_seeds // 2)]

    print(
        "Start Iteration...")  ################################################

    input_shape = model.n_features_
    block_size = int(np.cbrt(input_shape / dwi.shape[-1]))

    d = np.zeros([n_seeds, dwi.shape[-1] * block_size ** 3])
    dnorm = np.zeros([n_seeds, 1])
    vout = np.zeros([n_seeds, 3])
    for i in range(config['max_steps']):
        t0 = time()

        # Get coords of latest segement for each fiber
        ijk = xyz2ijk(xyz[:, -1, :], snap=True)

        n_ongoing = len(ijk)

        for ii, idx in enumerate(ijk):
            d[ii] = dwi[
                    idx[0] - (block_size // 2): idx[0] + (block_size // 2) + 1,
                    idx[1] - (block_size // 2): idx[1] + (block_size // 2) + 1,
                    idx[2] - (block_size // 2): idx[2] + (block_size // 2) + 1,
                    :].flatten()  # returns copy
            dnorm[ii] = np.linalg.norm(d[ii])
            d[ii] /= dnorm[ii]

        if i == 0:
            inputs = np.hstack([prior(xyz[:, 0, :]),
                                d[:n_ongoing], dnorm[:n_ongoing]])
        else:
            inputs = np.hstack([vout[:n_ongoing],
                                d[:n_ongoing], dnorm[:n_ongoing]])

        chunk = 2 ** 15  # 32768
        n_chunks = np.ceil(n_ongoing / chunk).astype(int)
        for c in range(n_chunks):

            outputs = model.predict(inputs[c * chunk: (c + 1) * chunk])
            v = bvecs[outputs, ...]
            vout[c * chunk: (c + 1) * chunk] = v

        rout = xyz[:, -1, :3] + config['step_size'] * vout
        rout = np.hstack([rout, np.ones((n_ongoing, 1))]).reshape(-1, 1, 4)

        xyz = np.concatenate([xyz, rout], axis=1)

        terminal_indices = terminator(xyz[:, -1, :])

        for idx in terminal_indices:
            gidx = fiber_idx[idx]
            # Other end not yet added
            if not fibers[gidx]:
                fibers[gidx].append(np.copy(xyz[idx, :, :3]))
            # Other end already added
            else:
                this_end = xyz[idx, :, :3]
                other_end = fibers[gidx][0]
                merged_fiber = np.vstack([
                    np.flip(this_end[1:], axis=0),
                    other_end])  # stitch ends together
                fibers[gidx] = [merged_fiber]

        xyz = np.delete(xyz, terminal_indices, axis=0)
        vout = np.delete(vout, terminal_indices, axis=0)
        fiber_idx = np.delete(fiber_idx, terminal_indices)

        print("Iter {:4d}/{}, finished {:5d}/{:5d} ({:3.0f}%) of all seeds with"
              " {:6.0f} steps/sec".format((i + 1), config['max_steps'],
                                          n_seeds - n_ongoing, n_seeds,
                                          100 * (1 - n_ongoing / n_seeds),
                                          n_ongoing / (time() - t0)),
              end="\r")

        if n_ongoing == 0:
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

    K.clear_session()
    if gpu_queue is not None:
        gpu_queue.put(gpu_idx)

    # Save Result

    fibers = [f[0] for f in fibers]

    tractogram = Tractogram(
        streamlines=ArraySequence(fibers),
        affine_to_rasmm=np.eye(4)
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    out_dir = os.path.join(os.path.dirname(config["dwi_path"]),
                           "predicted_fibers", timestamp)

    configs.deep_update(config, {"out_dir": out_dir})

    os.makedirs(out_dir, exist_ok=True)

    fiber_path = os.path.join(out_dir, timestamp + ".trk")
    print("\nSaving {}".format(fiber_path))
    TrkFile(tractogram, seed_file.header).save(fiber_path)

    config_path = os.path.join(out_dir, "config.yml")
    print("Saving {}".format(config_path))
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    if config["score"]:
        score_on_tm(fiber_path)

    return tractogram


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Use a trained model to predict fibers on DWI data.")

    parser.add_argument("config_path", type=str, nargs="?",
                        help="Path to inference config.")

    args, more_args = parser.parse_known_args()

    config = configs.compile_from(args.config_path, args, more_args)

    run_rf_inference(config)


