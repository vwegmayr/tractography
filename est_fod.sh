#!/bin/sh
# Usage: est_fod dwi_preproc.nii.gz bvecs bvals

# Uses input from preproc.sh

# Pipeline references:
# https://osf.io/ht7zv/ (Tutorial)
# https://mrtrix.readthedocs.io/en/latest/

# Check brain mask
# mrview dwi_denoised_pe_corr.mif -overlay.load dwi_brain_mask.mif

# Check response estimation:
# mrview dwi_denoised_pe_corr.mif -overlay.load response_voxels.mif
# shview response.txt

# Check fods with WM mask
# mrview wm.nii.gz -odf.load_sh fod_norm.mif


################################################################################

# Extract b=0 and b=1000 volumes
dwiextract $1 dwi_subset.mif \
-fslgrad $2 $3 \
-shells 0,1000 \
-force &&


# Brain Mask
dwi2mask dwi_subset.mif dwi_brain_mask.mif \
-force &&

# Response Function
dwi2response tournier dwi_subset.mif response.txt \
-voxels response_voxels.mif \
-mask dwi_brain_mask.mif \
-force &&

# FOD Estimation
dwi2fod csd dwi_subset.mif response.txt fod.mif \
-mask dwi_brain_mask.mif \
-lmax 4 \
-force &&

# Intensity Normalization
mtnormalise fod.mif fod_norm.mif \
-mask dwi_brain_mask.mif \
-force

################################################################################

# Brain Mask
dwi2mask dwi_denoised_pe_corr.mif dwi_brain_mask.mif \
-grad grad_corr \
-force &&

# Response Function
dwi2response tournier dwi_denoised_pe_corr.mif response.txt \
-voxels response_voxels.mif \
-mask dwi_brain_mask.mif \
-grad grad_corr \
-force &&

# FOD Estimation
dwi2fod csd dwi_denoised_pe_corr.mif response.txt fod.mif \
-lmax 4 \
-mask dwi_brain_mask.mif \
-grad grad_corr \
-force &&

# Intensity Normalization
mtnormalise fod.mif fod_norm.mif \
-mask dwi_brain_mask.mif \
-force