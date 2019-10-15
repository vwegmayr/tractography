#!/bin/bash
# Usage:
# ./unpack_hcp.sh "subjectID"

#unzip "hcp_zips/${1}_3T_Diffusion_preproc.zip"

dir="subjects/${1}"

mkdir -p $dir

shopt -s dotglob nullglob
mv "${1}/T1w/Diffusion/*" $dir

rm -r "${dir}/eddylogs"
rm "${dir}/grad_dev.nii.gz"