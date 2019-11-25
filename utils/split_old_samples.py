import os
import argparse
import numpy as np


def split_samples(samples, n_files=100):
    for sample_file in samples:
        out_dir = os.path.join(os.path.dirname(sample_file), 'splitted')
        os.makedirs(out_dir, exist_ok=True)
        sample_i = np.load(sample_file, allow_pickle=True)

        n_samples = sample_i['n_samples']
        input_shape = sample_i['input_shape']
        sample_path = os.path.join(out_dir, "samples-{0}.npz")
        n_per_file = n_samples // n_files

        for i in range(n_files):
            path_to_save = sample_path.format(i)
            print("Saving {}".format(path_to_save))

            if i == n_files - 1:
                sample_tosave = {'inputs': sample_i['inputs'][i * n_per_file:, ...],
                                 'outgoing': sample_i['outgoing'][i * n_per_file:,
                                             ...],
                                 'isterminal': sample_i['isterminal'][
                                               i * n_per_file:, ...]}
            else:
                sample_tosave = {'inputs': sample_i['inputs'][
                                           i * n_per_file: (i + 1) * n_per_file,
                                           ...],
                                 'outgoing': sample_i['outgoing'][
                                             i * n_per_file: (i + 1) * n_per_file,
                                             ...],
                                 'isterminal': sample_i['isterminal'][
                                               i * n_per_file: (i + 1) * n_per_file,
                                               ...]}

            np.savez(
                path_to_save,
                input_shape=input_shape,
                sample_shape=sample_tosave['inputs'].shape,
                n_samples=n_samples,
                **sample_tosave)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert old samples to the new version, by splitting them "
                    "to multiple files and add sample_shape attribute to each "
                    "file")

    parser.add_argument('samples', nargs='+', type=str,
                        help="list of paths to samples.npz")

    parser.add_argument("--n_files", default=100, type=int,
                        help="# of output files of the conditional samples")

    args = parser.parse_args()

    split_samples(args.samples, args.n_files)