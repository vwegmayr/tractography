import os
import argparse

import nibabel as nib
import numpy as np

from nibabel.streamlines.trk import TrkFile


def trim(trk_path, min_length, max_length):

    print("Loading fibers for trimming ...")
    trk_file = nib.streamlines.load(trk_path)

    tractogram = trk_file.tractogram

    print("Trimming fibers ...")
    keep = [i for i in range(len(tractogram))]
    for i, tract in enumerate(trk_file.tractogram):
        fiber = tract.streamline
        flen = np.linalg.norm(fiber[1:] - fiber[:-1], axis=1).sum()
        if flen < min_length or flen > max_length:
            keep.remove(i)
 
    trimmed_path = (
        trk_path[:-4] + "_{:2.0f}mm{:3.0f}.trk".format(
            min_length, max_length)
    )
    print("Saving trimmed fibers to {}".format(trimmed_path))
    TrkFile(tractogram[keep], trk_file.header).save(trimmed_path)

    return trimmed_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Trim fibers.")

    parser.add_argument("trk_path", type=str)

    parser.add_argument("--min_length", type=float, default=30)

    parser.add_argument("--max_length", type=float, default=200)

    args = parser.parse_args()

    trim(
        trk_path=args.trk_path,
        min_length=args.min_length,
        max_length=args.max_length
    )