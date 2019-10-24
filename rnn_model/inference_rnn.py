import time
import argparse

import nibabel as nib
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model


def run_inference(config):
    dwi_img = nib.load(config['dwi_path'])

    dwi_img = nib.funcs.as_closest_canonical(dwi_img)
    dwi_aff = dwi_img.affine
    dwi_affi = np.linalg.inv(dwi_aff)
    dwi = dwi_img.get_data()

    print("Loading Model...")

    model = load_model(config['model_path'])

    # Define coordinate transforms
    input_shape = model.layers[0].get_output_at(0).get_shape().as_list()[-1]
    block_size = int(np.cbrt(input_shape / dwi.shape[-1]))

    def xyz2ijk(coords, snap=False, shift=False):
        ijk = dwi_affi.dot(coords.T).T
        if snap:
            ijk = np.round(ijk).astype(int)
        return ijk


    print("Loading Seeds...")

    seed_file = nib.streamlines.load(config['seed_path'])
    seeds = seed_file.tractogram.streamlines.data
    seeds = np.vstack([seeds, seeds])  # Duplicate seeds for both directions
    seeds = np.hstack([seeds, np.ones([len(seeds), 1])])  # add affine dimension
    assert seeds.shape[-1] == 4  # (x, y, z, 1)
    print("Initialize Fibers...")

    xyz = seeds.reshape(-1, 1, 4)  # (fiber, segment, coord)

    fiber_idx = np.hstack([np.arange(len(seeds) // 2), np.arange(len(seeds) // 2)])
    fibers = [[] for _ in range(len(seeds) // 2)]

    print("Start Iteration...")
    max_steps = 100  # TODO
    for i in range(max_steps):
        t0 = time()

        ijk = xyz2ijk(xyz[:, -1, :], snap=True, shift=True)  # Get coords of latest segement for each fiber

        d = np.zeros([len(ijk), block_size, block_size, block_size, dwi.shape[-1]])

        for ii, idx in enumerate(ijk):
            d[ii] = dwi[idx[0] - (block_size // 2): idx[0] + (block_size // 2) + 1,
                    idx[1] - (block_size // 2): idx[1] + (block_size // 2) + 1,
                    idx[2] - (block_size // 2): idx[2] + (block_size // 2) + 1,
                    :]
        d = d.reshape(-1, dwi.shape[-1] * block_size ** 3)

        if i == 0:
            vin = prior(xyz[:, 0, :])
        else:
            vin = vout.copy()

        chunk_size = 2 ** 15  # 32768
        n_chunks = np.ceil(len(vin) / chunk_size).astype(int)

        inputs = np.hstack([vin, d])
        vout = np.zeros([len(vin), 3])
        for chunk in range(n_chunks):
            input_chunk = inputs[chunk * chunk_size: (chunk + 1) * chunk_size]
            if predict_fn == "mean":
                v = model(input_chunk).mean().numpy()
                v = normalize(v)  # Careful, the FvM mean is not a unit vector!
            else:
                v = model(input_chunk).sample().numpy()  # Samples are unit length, though!
            vout[chunk * chunk_size: (chunk + 1) * chunk_size] = v

        rout = (xyz[:, -1, :3] + step_size * vout)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Use a trained RNN model to predict fibers on DWI data.")

    parser.add_argument("model_path", type=str,
        help="Path to stored keras model.")

    parser.add_argument("dwi_path", type=str,
        help="Path to DWI data.")

    # parser.add_argument("prior_path", type=str,
    #     help="Path to prior file with either .nii or .h5 extension."
    #         "If .nii, assumes a WxLxHx3 volume containing the prior directions."
    #         "If .h5, assumes a trained model, which takes only DWI as input.")

    parser.add_argument("seed_path", type=str,
        help="Path to seed file (.trk).")

    # parser.add_argument("term_path", type=str,
    #     help="Path to terminator file (.nii).")

    # parser.add_argument("--thresh", type=float, default=0.1,
    #     help="Stopping threshold, used together with term_path, if provided.")

    # parser.add_argument("--predict_fn", type=str, default="mean",
    #     choices=["mean", "sample"],
    #     help="Next-Step prediction mode, either along most-likely direction "
    #     "(mean), or along a randomly sampled direction (sample).")

    # parser.add_argument("--step_size", type=float, default=0.25,
    #     help="Length of each step.")
    #
    # parser.add_argument("--max_steps", type=int, default=400,
    #     help="Maximum number of iterations.")
    #
    # parser.add_argument("--out_dir", type=str, default=None,
    #     help="Directory to save the predicted fibers. "
    #     "By default, it is created next to dwi_path.")

    args = parser.parse_args()

    config = dict(
        model_path=args.model_path,
        dwi_path=args.dwi_path,
        seed_path=args.seed_path
    )

    run_inference(config)