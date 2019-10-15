#!/bin/bash

# Basic ISMRM ##################################################################

dir=subjects/ismrm_basic

dwidenoise \
"${dir}/data.nii.gz" \
"${dir}/data_denoise.nii.gz" \
-extent 5,5,5 &&

dwigradcheck \
"${dir}/data_denoise.nii.gz" \
-fslgrad "${dir}/bvecs" "${dir}/bvals" \
-export_grad_fsl "${dir}/bvecs_check" "${dir}/bvals_check" &&

# Motion+Eddy+Distortion Correction
dwipreproc \
"${dir}/data_denoise.nii.gz" \
"${dir}/data_denoise_preproc.nii.gz" \
-rpe_none \
-pe_dir PA \
-fslgrad "${dir}/bvecs_check" "${dir}/bvals_check" \
-export_grad_fsl "${dir}/bvecs_preproc" "${dir}/bvals_preproc" \
-eddy_options " --slm=linear" \
-nthreads 20 &&

dwi2mask "${dir}/data_denoise_preproc.nii.gz" "${dir}/dwi_brain.nii.gz" \
-fslgrad "${dir}/bvecs_preproc" "${dir}/bvals_preproc" &&

dwinormalise \
"${dir}/data_denoise_preproc.nii.gz" \
"${dir}/dwi_brain.nii.gz" \
"${dir}/data_denoise_preproc_norm.nii.gz" \
-fslgrad "${dir}/bvecs_preproc" "${dir}/bvals_preproc"

# Ground Truth ISMRM ###########################################################

dir=subjects/ismrm_gt

dwigradcheck \
"${dir}/data.nii.gz" \
-fslgrad "${dir}/bvecs" "${dir}/bvals" \
-export_grad_fsl "${dir}/bvecs_check" "${dir}/bvals_check" &&

dwi2mask "${dir}/data.nii.gz" "${dir}/dwi_brain.nii.gz" \
-fslgrad "${dir}/bvecs_check" "${dir}/bvals_check"

dwinormalise \
"${dir}/data.nii.gz" \
"${dir}/dwi_brain.nii.gz" \
"${dir}/data_norm.nii.gz" \
-fslgrad "${dir}/bvecs_check" "${dir}/bvals_check"

# Reverse Phase ISMRM ##########################################################

# TODO