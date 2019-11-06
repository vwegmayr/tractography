import os
import gc
import argparse
import datetime
import logging
import yaml

import nibabel as nib
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from nibabel.streamlines.trk import TrkFile
from nibabel.streamlines.array_sequence import ArraySequence, concatenate
from nibabel.streamlines.tractogram import Tractogram

from GPUtil import getFirstAvailable
from sklearn.preprocessing import normalize
from time import time

from models import MODELS
from models import RNN as RNNModel

from utils.config import load

os.environ['PYTHONHASHSEED'] = '0'
tf.compat.v1.set_random_seed(42)
np.random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getFirstAvailable(
        order="load", maxLoad=10 ** -6, maxMemory=10 ** -1)[0])
except Exception as e:
    print(str(e))


class MarginHandler(object):

    def xyz2ijk(self, xyz):
        ijk = (xyz.T).copy()
        self.affi.dot(ijk, out=ijk)
        return np.round(ijk, out=ijk).astype(int, copy=False)


class Prior(MarginHandler):

    def __init__(self, prior_path):
        if ".nii" in prior_path:
            vec_img = nib.load(prior_path)
            self.vec = vec_img.get_data()
            self.affi = np.linalg.inv(vec_img.affine)
        elif ".h5" in prior_path:
            raise NotImplementedError # TODO: Implement prior model
        
    def __call__(self, xyz):
        if hasattr(self, "vec"):
            ijk = self.xyz2ijk(xyz)
            vecs = self.vec[ijk[0], ijk[1], ijk[2]] # fancy indexing -> copy!
            # Assuming that seeds have been duplicated for both directions!
            vecs[len(vecs)//2:, :] *= -1
            return normalize(vecs)
        elif hasattr(self, "model"):
            raise NotImplementedError # TODO: Implement prior model


class Terminator(MarginHandler):

    def __init__(self, term_path, thresh):
        if ".nii" in term_path:
            scalar_img = nib.load(term_path)
            self.scalar = scalar_img.get_data()
            self.affi = np.linalg.inv(scalar_img.affine)
        elif ".h5" in term_path:
            raise NotImplementedError # TODO: Implement termination model
        self.threshold = thresh

    def __call__(self, xyz):
        if hasattr(self, "scalar"):
            ijk = self.xyz2ijk(xyz)
            return np.where(
                self.scalar[ijk[0], ijk[1], ijk[2]] < self.threshold)[0]
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

    print("Loading DWI...") ####################################################

    dwi_img = nib.load(dwi_path)
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

    print("Loading Models...") #################################################

    config_path = os.path.join(os.path.dirname(model_path), "config.yml")

    model_name = load(config_path, "model_name")

    if hasattr(MODELS[model_name], "custom_objects"):
        model = load_model(model_path,
                           custom_objects=MODELS[model_name].custom_objects,
                           compile=False)
    else:
        model = load_model(model_path, compile=False)

    terminator = Terminator(term_path, thresh)

    prior = Prior(prior_path)

    print("Initializing Fibers...") ############################################

    seed_file = nib.streamlines.load(seed_path)
    xyz = seed_file.tractogram.streamlines.data
    n_seeds = 2 * len(xyz)
    xyz = np.vstack([xyz, xyz])  # Duplicate seeds for both directions
    xyz = np.hstack([xyz, np.ones([n_seeds, 1])]) # add affine dimension
    xyz = xyz.reshape(-1, 1, 4)  # (fiber, segment, coord)

    fiber_idx = np.hstack([
        np.arange(n_seeds//2, dtype="int32"),
        np.arange(n_seeds//2,  dtype="int32")
    ])
    fibers = [[] for _ in range(n_seeds//2)]

    print("Start Iteration...") ################################################

    input_shape = model.layers[0].get_output_at(0).get_shape().as_list()[-1]
    block_size = int(np.cbrt(input_shape / dwi.shape[-1]))

    d = np.zeros([n_seeds, dwi.shape[-1] * block_size**3])
    dnorm = np.zeros([n_seeds, 1])
    vout = np.zeros([n_seeds, 3])
    for i in range(max_steps):
        t0 = time()

        ijk = xyz2ijk(xyz[:,-1,:], snap=True)  # Get coords of latest segement for each fiber

        n_ongoing = len(ijk)

        for ii, idx in enumerate(ijk):
            d[ii] = dwi[idx[0] - (block_size // 2) : idx[0] + (block_size // 2) + 1,
                        idx[1] - (block_size // 2) : idx[1] + (block_size // 2) + 1,
                        idx[2] - (block_size // 2) : idx[2] + (block_size // 2) + 1,
                    :].flatten() # returns copy
            dnorm[ii] = np.linalg.norm(d[ii])
            d[ii] /= dnorm[ii]

        if i == 0:
            inputs = np.hstack([prior(xyz[:,0,:]), d[:n_ongoing], dnorm[:n_ongoing]])
        else:
            inputs = np.hstack([vout[:n_ongoing], d[:n_ongoing], dnorm[:n_ongoing]])

        chunk = 2**15 # 32768
        n_chunks = np.ceil(n_ongoing / chunk).astype(int)
        for c in range(n_chunks):

            outputs = model(inputs[c * chunk : (c + 1) * chunk])

            if isinstance(outputs, list):
                outputs = outputs[0]

            if predict_fn == "mean":
                v = outputs.mean_direction.numpy()
                # v = normalize(v)
            elif predict_fn == "sample":
                v = outputs.sample().numpy()
            vout[c * chunk : (c + 1) * chunk] = v

        rout = xyz[:, -1, :3] + step_size * vout
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
                    other_end]) # stitch ends together
                fibers[gidx] = [merged_fiber]

        xyz = np.delete(xyz, terminal_indices, axis=0)
        vout = np.delete(vout, terminal_indices, axis=0)
        fiber_idx = np.delete(fiber_idx, terminal_indices)

        print("Iter {:4d}/{}, finished {:5d}/{:5d} ({:3.0f}%) of all seeds with"
            " {:6.0f} steps/sec".format(
            (i+1), max_steps, n_seeds-n_ongoing, n_seeds,
            100*(1-n_ongoing/n_seeds), n_ongoing / (time() - t0)),
            end="\r"
        )

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

    # Save Result

    fibers = [f[0] for f in fibers]

    tractogram = Tractogram(
        streamlines=ArraySequence(fibers),
        affine_to_rasmm=np.eye(4)
    )

    if out_dir is None:
        out_dir = os.path.dirname(dwi_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    out_dir = os.path.join(out_dir, "predicted_fibers", timestamp)

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


def infere_batch_seed(xyz, prior, terminator, model, dwi, dwi_affi, max_steps, step_size):

    n_seeds = len(xyz) ## duplicated before, so multiple of 2
    fiber_idx = np.hstack([
        np.arange(n_seeds//2, dtype="int32"),
        np.arange(n_seeds//2,  dtype="int32")
    ])
    fibers = [[] for _ in range(n_seeds//2)]

    def xyz2ijk(coords, snap=False):
        ijk = (coords.T).copy()
        dwi_affi.dot(ijk, out=ijk)
        if snap:
            return np.round(ijk, out=ijk).astype(int, copy=False).T
        else:
            return ijk.T

    input_shape = model.layers[0].get_output_at(0).get_shape().as_list()[-1]
    block_size = int(np.cbrt(input_shape / dwi.shape[-1]))

    d = np.zeros([n_seeds, dwi.shape[-1] * block_size ** 3])
    dnorm = np.zeros([n_seeds, 1])
    vout = np.zeros([n_seeds, 3])
    already_terminated = np.empty(0, dtype="int32")
    mask = np.ones((n_seeds), dtype=bool)
    n_ongoing = n_seeds
    out_of_bound_fibers = 0
    for i in range(max_steps):
        t0 = time()

        ijk = xyz2ijk(xyz[:, -1, :], snap=True)  # Get coords of latest segement for each fiber

        for ii, idx in enumerate(ijk):
            try:
                d[ii] = dwi[idx[0] - (block_size // 2): idx[0] + (block_size // 2) + 1,
                        idx[1] - (block_size // 2): idx[1] + (block_size // 2) + 1,
                        idx[2] - (block_size // 2): idx[2] + (block_size // 2) + 1,
                        :].flatten()  # returns copy
                dnorm[ii] = np.linalg.norm(d[ii]) + 0.0000000001
                d[ii] /= dnorm[ii]
            except:
                assert ii in already_terminated
                out_of_bound_fibers = out_of_bound_fibers + 1

        if i == 0:
            inputs = np.hstack([prior(xyz[:, 0, :]), d, dnorm])
        else:
            inputs = np.hstack([vout, d, dnorm])

        vout = model.predict(inputs[:, np.newaxis, :]).squeeze()

        rout = xyz[:, -1, :3] + step_size * vout
        rout = np.hstack([rout, np.ones((n_seeds, 1))]).reshape(-1, 1, 4)

        xyz = np.concatenate([xyz, rout], axis=1)

        mask[already_terminated] = False
        tmp_indices = terminator(xyz[mask, -1, :])
        terminal_indices = np.where(mask)[0][tmp_indices]

        for idx in terminal_indices:
            assert idx not in already_terminated
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

            n_ongoing = n_ongoing - 1
        already_terminated = np.concatenate([already_terminated, terminal_indices])

        print("Iter {:4d}/{}, finished {:5d}/{:5d} ({:3.0f}%) of all seeds with"
              " {:6.0f} steps/sec".format(
            (i + 1), max_steps, n_seeds - n_ongoing, n_seeds,
                                100 * (1 - n_ongoing / n_seeds), n_ongoing / (time() - t0)),
            end="\r"
        )

        if n_ongoing == 0:
            assert len(set(already_terminated)) == n_seeds
            print("normal termination")
            break

        gc.collect()

    print("{0} times fibers got out of bound, but keep calm as they were already finished".format(out_of_bound_fibers))

    # Include unfinished fibers:
    for idx, gidx in enumerate(fiber_idx):
        if idx not in already_terminated:
            if not fibers[gidx]:
                fibers[gidx].append(xyz[idx, :, :3])
            else:
                this_end = xyz[idx, :, :3]
                other_end = fibers[gidx][0]
                merged_fiber = np.vstack([np.flip(this_end[1:], axis=0), other_end])
                fibers[gidx] = [merged_fiber]
            already_terminated = np.concatenate([already_terminated, [idx]])

    return fibers


def run_rnn_inference(
        model_path,
        dwi_path,
        prior_path,
        seed_path,
        term_path,
        thresh,
        predict_fn,
        step_size,
        max_steps,
        batch_size,
        out_dir
):
    """"""
    print("Loading DWI...")  ####################################################

    dwi_img = nib.load(dwi_path)
    dwi_img = nib.funcs.as_closest_canonical(dwi_img)
    dwi_aff = dwi_img.affine
    dwi_affi = np.linalg.inv(dwi_aff)
    dwi = dwi_img.get_data()

    print("Loading Models...")  #################################################

    config_path = os.path.join(os.path.dirname(model_path), "config.yml")

    with open(config_path, "r") as config_file:
        model_name = yaml.load(config_file)["model_name"]

    if hasattr(MODELS[model_name], "custom_objects"):
        trained_model = load_model(model_path,
                           custom_objects=MODELS[model_name].custom_objects)
    else:
        trained_model = load_model(model_path)

    model_config = {'batch_size': batch_size,
                    'input_shape':  trained_model.input_shape[1:]}
    prediction_model = RNNModel(model_config).keras
    prediction_model.set_weights(trained_model.get_weights())

    terminator = Terminator(term_path, thresh)

    prior = Prior(prior_path)

    print("Initializing Fibers...")  ############################################

    seed_file = nib.streamlines.load(seed_path)
    xyz = seed_file.tractogram.streamlines.data
    n_seeds = len(xyz)
    fibers = [[] for _ in range(n_seeds)]

    for i in range(0, n_seeds, batch_size // 2):
        xyz_batch = xyz[i:i + batch_size // 2]

        n_seeds_batch = 2 * len(xyz_batch)
        xyz_batch = np.vstack([xyz_batch, xyz_batch])  # Duplicate seeds for both directions
        xyz_batch = np.hstack([xyz_batch, np.ones([n_seeds_batch, 1])])  # add affine dimension
        xyz_batch = xyz_batch.reshape(-1, 1, 4)  # (fiber, segment, coord)

        # Make a last model for the remaining batch
        if i == batch_size//2 * (n_seeds // (batch_size // 2)):
            last_batch_size = (n_seeds - i) * 2
            model_config['batch_size'] = last_batch_size
            prediction_model = RNNModel(model_config).keras
            prediction_model.set_weights(trained_model.get_weights())

        prediction_model.reset_states()
        print("Batch {0} with shape {1}".format(i // (batch_size // 2), xyz_batch.shape))
        batch_fibers = infere_batch_seed(xyz_batch, prior, terminator,
            prediction_model, dwi, dwi_affi, max_steps, step_size)
        fibers[i:i+batch_size//2] = batch_fibers

    # Save Result
    fibers = [f[0] for f in fibers]

    tractogram = Tractogram(
        streamlines=ArraySequence(fibers),
        affine_to_rasmm=np.eye(4)
    )

    if out_dir is None:
        out_dir = os.path.dirname(dwi_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    out_dir = os.path.join(out_dir, "predicted_fibers", timestamp)

    os.makedirs(out_dir, exist_ok=True)

    fiber_path = os.path.join(out_dir, "fibers.trk")
    print("\nSaving {}".format(fiber_path))
    TrkFile(tractogram, seed_file.header).save(fiber_path)

    config = dict(
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

    parser.add_argument("--batch_size", type=int, default=512, help="Batch size of the model during predictions")

    parser.add_argument("--out_dir", type=str, default=None,
        help="Directory to save the predicted fibers. "
        "By default, it is created next to dwi_path.")

    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(args.model_path), "config.yml")

    if load(config_path, "model_name") == "RNN":
        run_rnn_inference(
            args.model_path,
            args.dwi_path,
            args.prior_path,
            args.seed_path,
            args.term_path,
            args.thresh,
            args.predict_fn,
            args.step_size,
            args.max_steps,
            args.batch_size,
            args.out_dir
        )
    else:
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


