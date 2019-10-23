#!/bin/bash

# Usage: ./preproc_ismrm.sh version
# where version is one of (basic, rpe, gt).
#
# Assumes you ran "./get_ismrm_data.sh version" before.
#
# Additional Requirements:
# MRtrix installation

dir="subjects/ismrm_$1"

# Basic ISMRM ##################################################################

if [ $1 = "basic" ]
then
    dwidenoise \
    "${dir}/data.nii.gz" \
    "${dir}/data_denoise.nii.gz" \
    -extent 5,5,5 &&

    dwigradcheck \
    "${dir}/data_denoise.nii.gz" \
    -fslgrad "${dir}/bvecs" "${dir}/bvals" \
    -export_grad_fsl "${dir}/bvecs_check" "${dir}/bvals_check" &&

    dwipreproc \
    "${dir}/data_denoise.nii.gz" \
    "${dir}/data_denoise_preproc.nii.gz" \
    -rpe_none \
    -pe_dir AP \
    -fslgrad "${dir}/bvecs_check" "${dir}/bvals_check" \
    -export_grad_fsl "${dir}/bvecs_input" "${dir}/bvals_input" \
    -eddy_options " --slm=linear" \
    -nthreads 20 &&

    dwi2mask \
    "${dir}/data_denoise_preproc.nii.gz" \
    "${dir}/dwi_brain_mask.nii.gz" \
    -fslgrad "${dir}/bvecs_input" "${dir}/bvals_input" &&

    dtifit \
    --data "${dir}/data_denoise_preproc.nii.gz" \
    --out "tensor" \
    --bvecs "${dir}/bvecs_input" \
    --bvals "${dir}/bvals_input" \
    --mask "${dir}/dwi_brain_mask.nii.gz" &&

    mrthreshold \
    "${dir}/fa.nii.gz" \
    "${dir}/dwi_wm_mask.nii.gz" \
    -mask "${dir}/dwi_brain_mask.nii.gz" &&

    dwinormalise \
    "${dir}/data_denoise_preproc.nii.gz" \
    "${dir}/dwi_wm_mask.nii.gz" \
    "${dir}/data_input.nii.gz" \
    -fslgrad "${dir}/bvecs_input" "${dir}/bvals_input" &&

    rm -r "dwi"*"-tmp-"*
fi

# Ground Truth ISMRM ###########################################################

if [ $1 = "gt" ]
then
    dwigradcheck \
    "${dir}/data.nii.gz" \
    -fslgrad "${dir}/bvecs" "${dir}/bvals" \
    -export_grad_fsl "${dir}/bvecs_input" "${dir}/bvals_input" &&

    dwi2mask \
    "${dir}/data.nii.gz" \
    "${dir}/dwi_brain_mask.nii.gz" \
    -fslgrad "${dir}/bvecs_input" "${dir}/bvals_input" &&

    dtifit \
    --data "${dir}/data.nii.gz" \
    --out "tensor" \
    --bvecs "${dir}/bvecs_input" \
    --bvals "${dir}/bvals_input" \
    --mask "${dir}/dwi_brain_mask.nii.gz" &&

    mrthreshold \
    "${dir}/fa.nii.gz" \
    "${dir}/dwi_wm_mask.nii.gz" \
    -mask "${dir}/dwi_brain_mask.nii.gz" &&

    dwinormalise \
    "${dir}/data.nii.gz" \
    "${dir}/dwi_wm_mask.nii.gz" \
    "${dir}/data_input.nii.gz" \
    -fslgrad "${dir}/bvecs_input" "${dir}/bvals_input"
fi

# Reverse Phase ISMRM ##########################################################

if [ $1 = "rpe" ]
then
    dwidenoise \
    "${dir}/data.nii.gz" \
    "${dir}/data_denoise.nii.gz" \
    -extent 5,5,5 &&

    dwigradcheck \
    "${dir}/data_denoise.nii.gz" \
    -fslgrad "${dir}/bvecs" "${dir}/bvals" \
    -export_grad_fsl "${dir}/bvecs_check" "${dir}/bvals_check" &&

    dwiextract \
    "${dir}/data_denoise.nii.gz" \
    "${dir}/b0.nii.gz" \
    -fslgrad "${dir}/bvecs_check" "${dir}/bvals_check" \
    -bzero &&

    # Extract volumes 0 and 2-33 along axis=3, i.e. remove the rpe volume.
    mrconvert "${dir}/data_denoise.nii.gz" \
    "${dir}/data_denoise_pe.nii.gz" \
    -coord 3 0,2:33 \
    -fslgrad "${dir}/bvecs_check" "${dir}/bvals_check" \
    -export_grad_fsl "${dir}/bvecs_check" "${dir}/bvals_check" \
    -force &&

    dwipreproc \
    "${dir}/data_denoise_pe.nii.gz" \
    "${dir}/data_denoise_preproc.nii.gz" \
    -rpe_pair \
    -pe_dir AP \
    -se_epi "${dir}/b0.nii.gz" \
    -fslgrad "${dir}/bvecs_check" "${dir}/bvals_check" \
    -export_grad_fsl "${dir}/bvecs_input" "${dir}/bvals_input" \
    -eddy_options " --slm=linear" \
    -nthreads 20 &&

    dwi2mask \
    "${dir}/data_denoise_preproc.nii.gz" \
    "${dir}/dwi_brain_mask.nii.gz" \
    -fslgrad "${dir}/bvecs_input" "${dir}/bvals_input" &&

    dtifit \
    --data "${dir}/data_denoise_preproc.nii.gz" \
    --out "tensor" \
    --bvecs "${dir}/bvecs_input" \
    --bvals "${dir}/bvals_input" \
    --mask "${dir}/dwi_brain_mask.nii.gz" &&

    mrthreshold \
    "${dir}/fa.nii.gz" \
    "${dir}/dwi_wm_mask.nii.gz" \
    -mask "${dir}/dwi_brain_mask.nii.gz" &&

    dwinormalise \
    "${dir}/data_denoise_preproc.nii.gz" \
    "${dir}/dwi_wm_mask.nii.gz" \
    "${dir}/data_input.nii.gz" \
    -fslgrad "${dir}/bvecs_input" "${dir}/bvals_input" &&

    rm -r "dwi"*"-tmp-"*
fi

# Same for all ISMRM versions ##################################################

# Keep only FA and V1
rm "tensor_L*.nii.gz" &&
rm "tensor_M*.nii.gz" &&
rm "tensor_S0.nii.gz" &&
rm "tensor_V2.nii.gz" &&
rm "tensor_V3.nii.gz" &&

mrresize \
"${dir}/tensor_FA.nii.gz" \
"${dir}/tensor_FA_125.nii.gz" \
-voxel 1.25 &&

mrresize \
"${dir}/tensor_V1.nii.gz" \
"${dir}/tensor_V1_125.nii.gz" \
-voxel 1.25 &&

mrresize \
"scoring/scoring_data/masks/wm.nii.gz" \
"${dir}/wm_mask_125.nii.gz" \
-voxel 1.25 &&

mrresize \
"scoring/scoring_data/masks/all_rois.nii.gz" \
"${dir}/all_rois_125.nii.gz" \
-voxel 1.25
