import os
import re
import yaml
import argparse
import numpy as np
import nibabel as nib

from multiprocessing import SimpleQueue, Process
from time import sleep

import tensorflow as tf
from tensorflow.keras import backend as K

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import (AveragePointwiseEuclideanMetric,
    ResampleFeature, distance_matrix)
from scipy.optimize import linear_sum_assignment
from dipy.tracking._utils import _mapping_to_voxel, _to_voxel_coordinates

from models.model_classes import FisherVonMises
from models import load_model

from resample_trk import add_tangent 
from utils.config import load
from utils.training import setup_env, maybe_get_a_gpu
from utils.prediction import get_blocksize
from utils._dispatch import get_gpus

from configs import save


@setup_env
def agreement(model_path, dwi_path_1, trk_path_1, dwi_path_2, trk_path_2,
    wm_path, matching_thresh, cluster_thresh, centroid_size, gpu_queue=None):

    try:
        gpu_idx = maybe_get_a_gpu() if gpu_queue is None else gpu_queue.get()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    except Exception as e:
        print(str(e))

    temperature = np.round(float(re.findall("T=(.*)\.h5", model_path)[0]), 6)
    model = load_model(model_path)

    print("Load data ...")

    dwi_img_1 = nib.load(dwi_path_1)
    dwi_img_1 = nib.funcs.as_closest_canonical(dwi_img_1)
    affine_1 = dwi_img_1.affine
    dwi_1 = dwi_img_1.get_data()

    dwi_img_2 = nib.load(dwi_path_2)
    dwi_img_2 = nib.funcs.as_closest_canonical(dwi_img_2)
    affine_2 = dwi_img_2.affine
    dwi_2 = dwi_img_2.get_data()

    wm_img = nib.load(wm_path)
    wm_data = wm_img.get_data()
    n_wm = (wm_data > 0).sum()

    img_shape = dwi_1.shape[:-1]
    voxsize = dwi_img_1.header["pixdim"][1]

    #---------------------------------------------------------------------------

    trk_file_1 = nib.streamlines.load(trk_path_1)
    tractogram_1 = trk_file_1.tractogram
    if "t" not in tractogram_1.data_per_point:
        tractogram_1 = add_tangent(tractogram_1)
    streamlines_1 = tractogram_1.streamlines

    trk_file_2 = nib.streamlines.load(trk_path_2)
    tractogram_2 = trk_file_2.tractogram
    if "t" not in tractogram_2.data_per_point:
        tractogram_2 = add_tangent(tractogram_2)
    streamlines_2 = tractogram_2.streamlines

    ############################################################################

    print("Clustering streamlines ...")

    feature = ResampleFeature(nb_points=centroid_size)

    qb = QuickBundles(
        threshold=cluster_thresh,
        metric=AveragePointwiseEuclideanMetric(feature)
    )

    bundles_1 = qb.cluster(streamlines_1)
    bundles_1.refdata = tractogram_1

    bundles_2 = qb.cluster(streamlines_2)
    bundles_2.refdata = tractogram_2

    print("Matching bundles ...")

    cost = distance_matrix(
        AveragePointwiseEuclideanMetric(),
        bundles_1.centroids,
        bundles_2.centroids
    )

    _, assignments = linear_sum_assignment(cost)

    matches = []
    for i, idx in enumerate(assignments):
        if cost[i, idx] < matching_thresh:
            matches.append((i, idx))

    print("Found {} matches".format(len(matches)))

    print("Computing bundle masks ...")

    all_directions_1 = []
    all_directions_2 = []
    all_ijk = []
    n=0
    for i, j in matches:
        counts_1, directions_1 = bundle_map(
            bundles_1.clusters[i], affine_1, img_shape)
        counts_2, directions_2 = bundle_map(
            bundles_2.clusters[j], affine_2, img_shape)

        overlap = np.logical_and(counts_1 > 0, counts_2 > 0)
        overlap = np.logical_and(overlap, wm_data > 0)

        all_ijk.append(np.argwhere(overlap))
        all_directions_1.append(directions_1[overlap])
        all_directions_2.append(directions_2[overlap])

        print("Computed pair {:4d}/{:4d}".format(n, len(matches)), end="\r")
        n+=1

    all_directions_1 = np.vstack(all_directions_1)
    all_directions_2 = np.vstack(all_directions_2)
    all_ijk = np.vstack(all_ijk)

    n_segments = all_ijk.shape[0]

    ############################################################################

    print("Computing agreement ...")

    block_size = get_blocksize(model, dwi_1.shape[-1])

    d_1 = np.zeros([
        n_segments,
        block_size,  block_size, block_size,
        dwi_1.shape[-1]
    ])
    d_2 = np.zeros([
        n_segments,
        block_size,  block_size, block_size,
        dwi_1.shape[-1]
    ])
    i,j,k = all_ijk.T
    for idx in range(block_size**3):
        ii,jj,kk = np.unravel_index(idx, (block_size, block_size, block_size))
        d_1[:, ii, jj, kk, :] = dwi_1[i+ii-1, j+jj-1, k+kk-1, :]
        d_2[:, ii, jj, kk, :] = dwi_2[i+ii-1, j+jj-1, k+kk-1, :]

    d_1 = d_1.reshape(-1, dwi_1.shape[-1] * block_size**3)
    d_2 = d_2.reshape(-1, dwi_2.shape[-1] * block_size**3)

    dnorm_1 = np.linalg.norm(d_1, axis=1, keepdims=True) + 10**-2
    dnorm_2 = np.linalg.norm(d_2, axis=1, keepdims=True) + 10**-2

    d_1 /= dnorm_1
    d_2 /= dnorm_2

    model_inputs_1 = np.hstack([all_directions_1, d_1, dnorm_1])
    model_inputs_2 = np.hstack([all_directions_2, d_2, dnorm_2])

    asum, amin, amean, amax = agreement_for(
        model,
        model_inputs_1,
        model_inputs_2
    )

    K.clear_session()
    if gpu_queue is not None:
        gpu_queue.put(gpu_idx)
    
    agreement = {"temperature": temperature}
    agreement["value"] = asum / n_wm
    agreement["min"] = amin
    agreement["mean"] = amean
    agreement["max"] = amax
    agreement["n_vox"] = n_segments
    agreement["n_wm"] = n_wm
    agreement["thresh"] = matching_thresh
    agreement["dwi_1"] = dwi_path_1
    agreement["trk_1"] = trk_path_1
    agreement["dwi_2"] = dwi_path_2
    agreement["trk_2"] = trk_path_2
    agreement["ideal"] = ideal_agreement(temperature)

    ############################################################################

    save(agreement,
        "agreement_T={}.yml".format(temperature),
        os.path.dirname(model_path)
    )


def logZ(kappa):
    expk2 = np.exp(- 2 * kappa)
    return np.log(2*np.pi) + kappa + np.log1p(- expk2) - np.log(kappa)


def ideal_agreement(T):
    return np.log(4*np.pi) + logZ(2/T) - 2*logZ(1/T)


def agreement_for(model, inputs1, inputs2):

    n_segments = len(inputs1)

    all_fvm_log_agreements = np.zeros(n_segments)

    chunk = 2**15  # 32768
    n_chunks = np.ceil(n_segments / chunk).astype(int)
    for c in range(n_chunks):

        fvm_pred_1, _ = model(
            inputs1[c * chunk : (c + 1) * chunk])

        fvm_pred_2, _ = model(
            inputs2[c * chunk : (c + 1) * chunk])

        all_fvm_log_agreements[c * chunk : (c + 1) * chunk] = (
            fvm_log_agreement(fvm_pred_1, fvm_pred_2)
        )

    all_fvm_log_agreements = np.maximum(0, all_fvm_log_agreements)

    return (
        all_fvm_log_agreements.sum(),
        all_fvm_log_agreements.min(),
        all_fvm_log_agreements.mean(),
        all_fvm_log_agreements.max()
    )


def bundle_map(bundle, affine, img_shape):

    lin_T, offset = _mapping_to_voxel(affine)
    counts = np.zeros(img_shape, 'int')
    directions = np.zeros(img_shape + (3,))

    for tract in bundle:
        inds = _to_voxel_coordinates(tract.streamline, lin_T, offset)
        i, j, k = inds.T
        counts[i, j, k] += 1
        directions[i, j, k] += tract.data_for_points["t"]

    directions /= (np.expand_dims(counts, -1) + 10**-6)

    directions /= (
        np.linalg.norm(directions, axis=-1, keepdims=True) + 10**-6
    )

    return counts, directions


def fvm_log_agreement(fvm1, fvm2):
    fvm12 = FisherVonMises(
        mean_direction=fvm1.mean_direction, # just a dummy, not used
        concentration=tf.norm(
            fvm1.mean_direction * fvm1.concentration[:, tf.newaxis] +
            fvm2.mean_direction * fvm2.concentration[:, tf.newaxis],
            axis=1)
    )
    return (
        np.log(4*np.pi) + 
        fvm12._log_normalization()
        - fvm1._log_normalization()
        - fvm2._log_normalization()
    )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Calculate agreement.")

    parser.add_argument("config_path", type=str, nargs="?")

    parser.add_argument("--wm_path", help="Path to .nii file", type=str)

    parser.add_argument("--dwi_path_1", help="Path to .nii file", type=str)

    parser.add_argument("--dwi_path_2", help="Path to .nii file", type=str)

    parser.add_argument("--trk_path_1", help="Path to .trk file", type=str)

    parser.add_argument("--trk_path_2", help="Path to .trk file", type=str)

    parser.add_argument("--model_path", help="Path to .h5 file", type=str)

    parser.add_argument("--mthresh", help="Bundle matching threshold",
        type=float, default=20., dest="matching_thresh")

    parser.add_argument("--cthresh", help="Bundle clustering threshold",
        type=float, default=20., dest="cluster_thresh")

    parser.add_argument("--centroid_size", help="Length of fiber centroids",
        type=int, default=200)

    args = parser.parse_args()

    if args.config_path is not None:

        config = load(args.config_path)

        gpu_queue = SimpleQueue()
        for idx in get_gpus():
            gpu_queue.put(str(idx))

        try:
            procs=[]
            for model_path, pair in config["pred_pairs"].items():
                while gpu_queue.empty():
                    sleep(2)
                p = Process(
                    target=agreement,
                    args=(model_path,
                          pair[0]["dwi_path"],
                          pair[0]["trk_path"],
                          pair[1]["dwi_path"],
                          pair[1]["trk_path"],
                          config["wm_path"],
                          config["matching_thresh"],
                          config["cluster_thresh"],
                          config["centroid_size"],
                          gpu_queue)
                )
                procs.append(p)
                p.start()
                sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            for p in procs:
                p.join()
                while p.exitcode is None:
                    sleep(0.1)
    else:
        agreement(
            args.model_path,
            args.dwi_path_1,
            args.trk_path_1,
            args.dwi_path_2,
            args.trk_path_2,
            args.wm_path,
            args.matching_thresh,
            args.cluster_thresh,
            args.centroid_size
        )