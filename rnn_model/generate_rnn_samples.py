import time
import os
import random

import nibabel as nib
import numpy as np
import yaml
import argparse

from hashlib import md5
from scipy.interpolate import RegularGridInterpolator

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)


def create_samples(config):
    """"""
    if config["block_size"] % 2 == 0:
        raise ValueError("block_size must be an odd number (1, 3, 5,...)")

    if config["reverse_samples"] and config["max_n_samples"] % 2 != 0:
        raise ValueError("max_n_samples can not be an odd number for reverse_samples == True.")

    trk_path = config['trk_path']
    dwi_path = config['dwi_path']

    hasher = md5()
    for v in config.values():
        hasher.update(str(v).encode())

    tracts = nib.streamlines.load(trk_path).tractogram  # fiber coordinates in rasmm
    assert tracts.data_per_point is not None
    assert "t" in tracts.data_per_point

    dwi_img = nib.load(dwi_path)
    dwi_img = nib.funcs.as_closest_canonical(dwi_img)

    inv_affine = np.linalg.inv(dwi_img.affine)
    xyz2ijk = lambda r: inv_affine.dot([r[0], r[1], r[2], 1])[:3]

    dwi = dwi_img.get_data()

    n_fibers = len(tracts)
    fiber_lengths = [len(f) for f in tracts]
    n_samples = np.sum(fiber_lengths) - 2 * n_fibers
    if config["reverse_samples"]:
        n_samples *= 2
    n_samples = min(n_samples, config["max_n_samples"])
    print('number of samples: {0}'.format(n_samples))

    np.random.seed(42)
    perm = np.random.permutation(len(tracts))
    tracts = tracts[perm]

    inputs = []
    outputs = []
    total_samples = 0
    done = False
    for fi, f in enumerate(tracts):
        start = time.time()
        fib_len = len(f) - 2
        split_num = fib_len // config['seq_len']
        skip_num = fib_len % config['seq_len']

        split_num = split_num * 2 if config["reverse_samples"] else split_num
        skip_num = skip_num * 2 if config["reverse_samples"] else skip_num
        #         n_samples = n_samples - skip_num

        max_samples = split_num // 2 if config["reverse_samples"] else split_num
        tract_input = np.zeros((split_num, config['seq_len'], config['input_size']))
        tract_output = np.zeros((split_num, config['seq_len'], config['output_size']))
        for i, r in enumerate(f.streamline[1:-1]):  # Exclude end points
            try:
                idx = xyz2ijk(r)  # anchor idx
                IDX = np.round(idx).astype(int)

                values = np.zeros([3, 3, 3,
                                   config["block_size"], config["block_size"], config["block_size"],
                                   dwi.shape[-1]])

                for x in range(config["block_size"]):
                    for y in range(config["block_size"]):
                        for z in range(config["block_size"]):
                            values[x, y, z, :] = dwi[
                                                 IDX[0] + x - 2 * (config["block_size"] // 2): IDX[0] + x + 1,
                                                 IDX[1] + y - 2 * (config["block_size"] // 2): IDX[1] + y + 1,
                                                 IDX[2] + z - 2 * (config["block_size"] // 2): IDX[2] + z + 1,
                                                 :]
                fn = RegularGridInterpolator(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), values)

                d = fn([idx[0] - IDX[0], idx[1] - IDX[1], idx[2] - IDX[2]])[0]
                d = d.flatten()  # to get back the spatial order: reshape(bs, bs, bs, dwi.shape[-1])

            except IndexError:
                n_samples -= (2 if config["reverse_samples"] else 1)
                print("Index error at r={}, idx={}, fiber_idx={}\n".format(r, idx, perm[fi]) +
                      "Maybe wrong reference frame, or resampling failed."
                      )
                continue

            vout = f.data_for_points["t"][i + 1].astype("float32")
            vin = f.data_for_points["t"][i].astype("float32")

            i_pos = i // config['seq_len']
            j_pos = i % config['seq_len']

            if (i_pos == max_samples):
                break
            tract_input[i_pos, j_pos, ...] = np.hstack([vin, d]).astype("float32")
            tract_output[i_pos, j_pos, ...] = vout
            total_samples = total_samples + 1

            if config["reverse_samples"]:
                i_pos = (split_num // 2) + (i // config['seq_len'])
                j_pos = i % config['seq_len']

                tract_input[i_pos, j_pos, ...] = np.hstack([-vout, d]).astype("float32")
                tract_output[i_pos, j_pos, ...] = -vin
                total_samples = total_samples + 1

            if total_samples >= n_samples:
                done = True
                break

        inputs.append(tract_input)
        outputs.append(tract_output)
        print("Finished {:3.0f}% in {}".format(100 * total_samples / n_samples, time.time() - start), end="\r")

        if done:
            break

    out_dir = config['out_dir']
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(dwi_path), "samples")

    if os.path.exists(out_dir):
        print("Samples with this config have been created already:\n{}".format(out_dir))
        return None, None
    os.makedirs(out_dir)

    inputs = np.concatenate(inputs, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    save_path = os.path.join(out_dir, "samples.npz")

    print("Saving {}".format(save_path))
    np.savez_compressed(save_path, inputs=inputs, outputs=outputs)

    config["n_samples"] = int(n_samples)
    config_path = os.path.join(out_dir, "config" + ".yml")
    print("Saving {}".format(config_path))
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    return inputs, outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate sample npz from DWI and TRK data.")

    parser.add_argument("dwi_path", help="Path to DWI file. e.g. ../subjects/992774/fod_norm.nii.gz")

    parser.add_argument("trk_path",
                        help="Path to TRK file. e.g. ../subjects/992774/merged_tracks/merged_W020_s=5_n=auto.trk")

    parser.add_argument("--seq_len", help="Length of the RNN sequence",
        default=3, type=int)

    parser.add_argument("--input_size",
                        help="Size of the input features. This must be known in advance for memory allocation",
                        default=408, type=int)

    parser.add_argument("--output_size",
                        help="Size of the outputs. This must be known in advance for memory allocation",
                        default=3, type=int)

    parser.add_argument("--block_size", help="Size of cubic neighborhood.",
        default=3, choices=[1,3,5,7], type=int)

    parser.add_argument("--no_reverse", action="store_false",
        help="Do not include direction-reversed samples.")

    parser.add_argument("--keep_n", default=2**30, type=int,
        help="Maximum number of samples to keep.")

    parser.add_argument("--out_dir", default=None,
        help="Sample directory, by default creates directory next to dwi_path.")

    args = parser.parse_args()

    config = dict(
        sample_type="rnn",
        seq_len=args.seq_len,
        input_size=args.input_size,
        output_size=args.output_size,
        dwi_path=args.dwi_path,
        trk_path=args.trk_path,
        block_size=args.block_size,
        reverse_samples=False if args.no_reverse else True,
        max_n_samples=args.keep_n,
        out_dir=args.out_dir
    )

    create_samples(config)
