import os
import argparse

import nibabel as nib
import numpy as np
import subprocess

from nibabel.streamlines.trk import TrkFile
from pathos.pools import ProcessPool

def trim(trk_path, min_length, max_length, fast=True, overwrite=False):

    if overwrite:
        trimmed_path = trk_path
    else:
        trimmed_path = (
            trk_path[:-4] + "_{:2.0f}mm{:3.0f}.trk".format(
                min_length, max_length)
        )

    print("Loading fibers for trimming ...")
    trk_file = nib.streamlines.load(trk_path)
    tractogram = trk_file.tractogram

    print("Trimming fibers ...")
    if fast and len(tractogram.data_per_point) == 0:
        cmd = [
            "track_vis", trk_path, "-nr", "-l", str(min_length),
            str(max_length), "-o", trimmed_path
        ]
        cmd = " ".join(cmd)
        subprocess.run(cmd, shell=True)
    else:
        pool = ProcessPool(nodes=10)

        def has_ok_length(f):
            l = np.linalg.norm(f[1:] - f[:-1], axis=1).sum()
            if l < min_length or l > max_length:
                return False
            else:
                return True

        is_ok = pool.map(has_ok_length, tractogram.streamlines)

        TrkFile(tractogram[is_ok], trk_file.header).save(trimmed_path)

    print("Saving trimmed fibers to {}".format(trimmed_path))
    return trimmed_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Trim fibers.")

    parser.add_argument("trk_path", type=str)

    parser.add_argument("--min_length", type=int, default=30)

    parser.add_argument("--max_length", type=int, default=200)

    args = parser.parse_args()

    trim(
        trk_path=args.trk_path,
        min_length=args.min_length,
        max_length=args.max_length
    )