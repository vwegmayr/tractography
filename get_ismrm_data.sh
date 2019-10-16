#!/bin/bash

baseurl="http://www.tractometer.org/downloads/downloads/"

mkdir -p "subjects"

# Basic ISMRM ##################################################################

folder="ISMRM_2015_Tracto_challenge_data"
dir="subjects/ismrm_basic"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
wget "$baseurl${folder}_v1_1.zip"
elif [[ "$OSTYPE" == "darwin"* ]]; then
curl "$baseurl${folder}_v1_1.zip" -o ${folder}_v1_1.zip
fi

unzip "${folder}_v1_1.zip"

mv $folder $dir

rm "${folder}_v1_1.zip"

mv "${dir}/Diffusion.bvals" "${dir}/bvals"
mv "${dir}/Diffusion.bvecs" "${dir}/bvecs"
mv "${dir}/Diffusion.nii.gz" "${dir}/data.nii.gz"

# Reverse Phase ISMRM ##########################################################

folder=ISMRM_2015_Tracto_challenge_data_with_reversed_phase
dir="subjects/ismrm_rpe"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
wget "$baseurl${folder}_v1_0.zip"
elif [[ "$OSTYPE" == "darwin"* ]]; then
curl "$baseurl${folder}_v1_0.zip" -o ${folder}_v1_0.zip
fi

unzip "${folder}_v1_0.zip"

mv $folder $dir

rm "${folder}_v1_0.zip"

mv "${dir}/Diffusion_WITH_REVERSEPHASE.bvals" "${dir}/bvals"
mv "${dir}/Diffusion_WITH_REVERSEPHASE.bvecs" "${dir}/bvecs"
mv "${dir}/Diffusion_WITH_REVERSEPHASE.nii.gz" "${dir}/data.nii.gz"

# Ground Truth ISMRM ###########################################################

folder=ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2
dir="subjects/ismrm_gt"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
wget "${baseurl}ismrm_challenge_2015/${folder}.zip"
elif [[ "$OSTYPE" == "darwin"* ]]; then
curl "${baseurl}ismrm_challenge_2015/${folder}.zip" -o ${folder}.zip
fi

unzip "${folder}.zip"

mv $folder $dir

rm "${folder}.zip"

mv "${dir}/NoArtifacts_Relaxation.bvals" "${dir}/bvals"
mv "${dir}/NoArtifacts_Relaxation.bvecs" "${dir}/bvecs"
mv "${dir}/NoArtifacts_Relaxation.nii.gz" "${dir}/data.nii.gz"
