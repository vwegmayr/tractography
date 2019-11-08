import os
os.environ['PYTHONHASHSEED'] = '0'
import random as rn
rn.seed(12345)
from numpy.random import seed
seed(42)

import yaml
import argparse

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from hashlib import md5
from nibabel.streamlines.tractogram import Tractogram, PerArraySequenceDict
from nibabel.streamlines.trk import TrkFile
from nibabel.streamlines.array_sequence import ArraySequence


def fiber_geometry(fiber, npts, smoothing, higher_order):
    """Resample one fiber, and calculate geometry data"""
    
    tck, u = interpolate.splprep(fiber.T.reshape(3, -1), s=smoothing)

    if npts == "auto":
        flen = np.linalg.norm(fiber[1:] - fiber[:-1], axis=1).sum()
        npts = int(flen/2) # if units are in mm, pts are on average 2 mm apart
        pts = np.linspace(0, 1, npts)
    elif npts == "same":
        npts = len(u)
        pts=u
    else:
        pts = np.linspace(0, 1, npts)

    r = np.dstack(interpolate.splev(pts, tck))[0] # position
    r1 = np.dstack(interpolate.splev(pts, tck, der=1))[0]
    r2 = np.dstack(interpolate.splev(pts, tck, der=2))[0]
    r3 = np.dstack(interpolate.splev(pts, tck, der=3))[0]

    r1xr2 = np.cross(r1, r2)

    t = r1
    t /= np.linalg.norm(t, axis=1, keepdims=True) # tangent vector

    if higher_order:

        b = r1xr2
        b /= np.linalg.norm(b, axis=1, keepdims=True) # binormal vector

        n = np.cross(b, t) # main normal vector

        k = np.linalg.norm(r1xr2, axis=1, keepdims=True)
        k /= np.linalg.norm(r1, axis=1, keepdims=True)**3 # curvature

        tau = np.sum(r1xr2 * r3, axis=1, keepdims=True)
        tau /= np.linalg.norm(r1xr2, axis=1, keepdims=True)**2 # torsion
    
        return r, t, b, n, k, tau, npts

    else:

        return r, t, npts


def resample(trk_path, npts, smoothing, higher_order, out_dir):
    
    trk_file = nib.streamlines.load(trk_path)
    fibers = trk_file.tractogram.streamlines
    
    position = ArraySequence()
    tangent = ArraySequence()
    if higher_order:
        binormal = ArraySequence()
        mainnormal = ArraySequence()
        curvature = ArraySequence()
        torsion = ArraySequence() 
    rows = 0
    
    def max_dist_from_mean(path):
        return np.linalg.norm(path - np.mean(path, axis=0, keepdims=True),
            axis=1).max()
    
    n_fails = 0
    for i, f in enumerate(fibers):

        if len(f) < 4: # Too short to compute higher derivatives
            n_fails += 1
            continue

        if higher_order:
            r, t, b, n, k, tau, cnt = fiber_geometry(
                f,
                npts=npts,
                smoothing=smoothing,
                higher_order=higher_order)
        else:
            r, t, cnt = fiber_geometry(
                f,
                npts=npts,
                smoothing=smoothing,
                higher_order=higher_order)

        if max_dist_from_mean(r) > 1.2 * max_dist_from_mean(f):
            n_fails += 1
            continue
        
        position.append(r, cache_build=True)
        tangent.append(t, cache_build=True)
        if higher_order:
            binormal.append(b, cache_build=True)
            mainnormal.append(n, cache_build=True)
            curvature.append(k, cache_build=True)
            torsion.append(tau, cache_build=True)
        rows += cnt
        
        print("Finished {:3.0f}%".format(100*(i+1)/len(fibers)), end="\r")
    
    if n_fails > 0:
        print("Failed to resample {} out of {} ".format(n_fails, len(fibers)) +
            "fibers, they were not included.")
        
    position.finalize_append()
    tangent.finalize_append()
    if higher_order:
        binormal.finalize_append()
        mainnormal.finalize_append()
        curvature.finalize_append()
        torsion.finalize_append()
    
    if higher_order:
        data_per_point = PerArraySequenceDict(
            n_rows = rows,
            t = tangent,
            b = binormal,
            n = mainnormal,
            k = curvature,
            tau = torsion
        )
    else:
        data_per_point = PerArraySequenceDict(
            n_rows = rows,
            t = tangent,
        )
    
    tractogram = Tractogram(
        streamlines=position,
        data_per_point=data_per_point,
        affine_to_rasmm=np.eye(4) # Fiber coordinates are already in rasmm space!
    )
    
    if out_dir is None:
        out_dir = os.path.dirname(trk_path)
    
    basename = os.path.basename(trk_path).split(".")[0]
    save_path = os.path.join(out_dir, "{}_s={}_n={}.trk".format(
        basename, smoothing, npts))

    os.makedirs(out_dir, exist_ok=True)
    print("Saving {}".format(save_path))
    TrkFile(tractogram, trk_file.header).save(save_path)
    
    return tractogram


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Resample streamlines, and "
        "compute local geometry data."
        )

    parser.add_argument("trk_path", help="Path to .trk file", type=str)

    parser.add_argument("--npts", default="auto", # or "same"
        help="Number of resampling points, i.e. fiber length in points."
        )
    parser.add_argument("--smoothing", default=5, type=float,
        help="Amount of spatial smoothing, larger values mean more smoothing."
        )
    parser.add_argument("--higher_order", action="store_true",
        help="Include higher order derivatives."
        )
    parser.add_argument("--out_dir",
        help="Directory for saving the resampled .trk file.",
        type=str, default=None)
    args = parser.parse_args()

    resample(
        args.trk_path,
        args.npts,
        args.smoothing,
        args.higher_order,
        args.out_dir
    )