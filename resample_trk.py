import os
os.environ['PYTHONHASHSEED'] = '0'
import random as rn
rn.seed(12345)
from numpy.random import seed
seed(42)

import argparse

import numpy as np
import nibabel as nib

from scipy import interpolate
from nibabel.streamlines.tractogram import Tractogram, PerArraySequenceDict
from nibabel.streamlines.trk import TrkFile
from nibabel.streamlines.array_sequence import ArraySequence


def fiber_geometry(fiber, npts, smoothing):
    """Resample one fiber, and calculate geometry data"""
    
    tck, u = interpolate.splprep(fiber.T.reshape(3, -1), s=smoothing)

    if npts == "auto":
        flen = np.linalg.norm(fiber[1:] - fiber[:-1], axis=1).sum()
        npts = int(flen/2) # if units are in mm, pts are on average 2 mm apart
        pts = np.linspace(0, 1, npts)
    elif npts == "same":
        npts = len(u)
        pts = u * 1.0
    else:
        pts = np.linspace(0, 1, npts)

    r = np.dstack(interpolate.splev(pts, tck))[0] # position
    r1 = np.dstack(interpolate.splev(pts, tck, der=1))[0]

    t = r1 / np.linalg.norm(r1, axis=1, keepdims=True) # tangent vector

    return r, t, npts


def add_tangent(tractogram, min_length=0, max_length=1000):
    print("Adding tangents ...")
    return resample_tractogram(tractogram, npts="same", smoothing=0,
        min_length=min_length, max_length=max_length)


def resample_tractogram(tractogram, npts, smoothing,
    min_length=0, max_length=1000):

    streamlines = tractogram.streamlines

    position = ArraySequence()
    tangent = ArraySequence()
    rows = 0
    
    def max_dist_from_mean(path):
        return np.linalg.norm(path - np.mean(path, axis=0, keepdims=True),
            axis=1).max()
    
    n_fails = 0
    n_length = 0
    for i, f in enumerate(streamlines):

        flen = np.linalg.norm(f[1:] - f[:-1], axis=1).sum()
        if (flen < min_length) or (flen > max_length):
            n_length += 1
            continue

        r, t, cnt = fiber_geometry(f, npts=npts, smoothing=smoothing)
 
        if max_dist_from_mean(r) > 1.2 * max_dist_from_mean(f):
            n_fails += 1
            continue
        
        position.append(r, cache_build=True)
        tangent.append(t, cache_build=True)
        rows += cnt
        
        print("Finished {:3.0f}%".format(100*(i+1)/len(streamlines)), end="\r")
    
    if n_fails > 0:
        print("Failed to resample {} out of {} ".format(n_fails,
            len(streamlines)) + "fibers, they were not included.")
        
    if n_length > 0:
        print("{} out of {} ".format(n_length,
            len(streamlines)) + "fibers excluded by length.")

    position.finalize_append()
    tangent.finalize_append()

    other_data={}
    if npts == "same":
        for key in list(tractogram.data_per_point.keys()):
            if key != "t":
                other_data[key] = tractogram.data_per_point[key]

    data_per_point = PerArraySequenceDict(
        n_rows = rows,
        t = tangent,
        **other_data
    )

    return Tractogram(
        streamlines=position,
        data_per_point=data_per_point,
        affine_to_rasmm=np.eye(4) # Fiber coordinates are already in rasmm space!
    )


def resample(trk_path, npts, smoothing, out_dir, min_length=0, max_length=1000):
    
    trk_file = nib.streamlines.load(trk_path)

    tractogram = resample_tractogram(trk_file.tractogram, npts, smoothing,
        min_length, max_length)
    
    if out_dir is None:
        out_dir = os.path.dirname(trk_path)
    
    basename = os.path.basename(trk_path).split(".")[0]
    save_path = os.path.join(out_dir, "{}_s={}_n={}.trk".format(
        basename, smoothing, npts))

    os.makedirs(out_dir, exist_ok=True)
    print("Saving {}".format(save_path))
    TrkFile(tractogram, trk_file.header).save(save_path)
    
    return tractogram


def maybe_add_tangent(trk_path, min_length=0, max_length=1000):

    cache_path = trk_path[:-4] + "_t.trk"

    if os.path.exists(cache_path):
        trk_file = nib.streamlines.load(cache_path)
        return trk_file.tractogram
    else:
        trk_file = nib.streamlines.load(trk_path)
        tractogram = trk_file.tractogram

        if "t" not in tractogram.data_per_point:
            tractogram = add_tangent(tractogram, min_length=min_length,
                max_length=max_length)
            TrkFile(tractogram, trk_file.header).save(cache_path)

        return tractogram


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Resample streamlines, and compute local geometry data.")
    parser.add_argument("trk_path", help="Path to .trk file", type=str)

    parser.add_argument("--npts", default="auto",  # or "same"
        help="Number of resampling points, i.e. fiber length in points."
        )
    parser.add_argument("--smoothing", default=5, type=float,
        help="Amount of spatial smoothing, larger values mean more smoothing."
        )
    parser.add_argument("--out_dir",
        help="Directory for saving the resampled .trk file.",
        type=str, default=None)
    args = parser.parse_args()

    resample(
        args.trk_path,
        args.npts,
        args.smoothing,
        args.out_dir
    )