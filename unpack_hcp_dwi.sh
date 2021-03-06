#!/bin/bash

# Usage: ./unpack_hcp.sh subjectID
# Example: ./unpack_hcp.sh 917255
#
# Assumes you manually downloaded a file like "917255_3T_Diffusion_preproc.zip"
# from https://db.humanconnectome.org to a folder named "hcp_zips". 
# Accessing this data online requires only free registration.
#
# Additional Requirements:
# MRtrix installation

dir="subjects/${1}"

if [ ! -d "subjects" ]
then
    mkdir "subjects"
fi

if [ ! -d "${dir}" ]
then
    mkdir "${dir}"
fi

# Unpack HCP DWI ###############################################################

unzip "hcp_zips/${1}_3T_Diffusion_preproc.zip" &&

subjectID=$(grep -Po "[0-9]+" <<< $1)

mv "${subjectID}/T1w/Diffusion/"* "${dir}" &&

rm -r "${dir}/eddylogs" &&
rm "${dir}/grad_dev.nii.gz" &&

rm -r ${subjectID} &&

# Perform basic preprocessing ##################################################

# Extract only b=0,1000 volumes
dwiextract \
"${dir}/data.nii.gz" \
"${dir}/data_1k.nii.gz" \
-fslgrad "${dir}/bvecs" "${dir}/bvals" \
-export_grad_fsl "${dir}/bvecs_1k" "${dir}/bvals_1k" \
-shells 0,1000 &&

dwigradcheck \
"${dir}/data_1k.nii.gz" \
-fslgrad "${dir}/bvecs_1k" "${dir}/bvals_1k" \
-export_grad_fsl "${dir}/bvecs_input" "${dir}/bvals_input" &&

dwi2mask "${dir}/data_1k.nii.gz" \
"${dir}/dwi_brain_mask.nii.gz" \
-fslgrad "${dir}/bvecs_input" "${dir}/bvals_input" &&

dtifit \
-k "${dir}/data_1k.nii.gz" \
-o "${dir}/tensor" \
-r "${dir}/bvecs_input" \
-b "${dir}/bvals_input" \
-m "${dir}/dwi_brain_mask.nii.gz" &&

mrthreshold \
"${dir}/tensor_FA.nii.gz" \
"${dir}/dwi_wm_mask.nii.gz" \
-mask "${dir}/dwi_brain_mask.nii.gz" &&

dwinormalise \
"${dir}/data_1k.nii.gz" \
"${dir}/dwi_wm_mask.nii.gz" \
"${dir}/data_input.nii.gz" \
-fslgrad "${dir}/bvecs_input" "${dir}/bvals_input" &&

# Keep only FA and V1
rm "${dir}/tensor_L"*".nii.gz" &&
rm "${dir}/tensor_M"*".nii.gz" &&
rm "${dir}/tensor_S0.nii.gz" &&
rm "${dir}/tensor_V2.nii.gz" &&
rm "${dir}/tensor_V3.nii.gz"
