import numpy as np
import nibabel as nib
from sklearn.preprocessing import normalize

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