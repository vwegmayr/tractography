#!/bin/bash

# Usage: ./est_fod subjectID
# Example: ./est_fod 917255

dir="subjects/$1"

dwi2response \
tournier \
"${dir}/data_input.nii.gz" \
"${dir}/response.txt" \
-mask "${dir}/dwi_brain_mask.nii.gz" \
-fslgrad "${dir}/bvecs_input" "${dir}/bvals_input" &&

dwi2fod \
csd \
"${dir}/data_input.nii.gz" \
"${dir}/response.txt" \
"${dir}/fod.nii.gz" \
-lmax 4 \
-mask "${dir}/dwi_brain_mask.nii.gz" \
-fslgrad "${dir}/bvecs_input" "${dir}/bvals_input" &&

mtnormalise \
"${dir}/fod.nii.gz" \
"${dir}/fod_norm.nii.gz" \
-mask "${dir}/dwi_brain_mask.nii.gz" &&

sh2peaks \
"${dir}/fod_norm.nii.gz" \
"${dir}/peaks.nii.gz" \
-num 1 &&

if [[ $dir =~ "ismrm" ]]
then
    mrresize \
    "${dir}/fod_norm.nii.gz" \
    "${dir}/fod_norm_125.nii.gz" \
    -voxel 1.25

    mrresize \
    "${dir}/fod_norm.nii.gz" \
    "${dir}/fod_norm_050.nii.gz" \
    -voxel 0.5
fi
