#!/bin/bash

# Usage: ./get_ismrm_data.sh version
# where version is one of (basic, rpe, gt).
#
# This script will create the folder subjects/ismrm_$version.
#
# The folder contains the respective version of the Tractometer diffusion data:
# basic: image includes noise and artefacts
# rpe: basic + reversed-phase-encoded (rpe) b0 image, for distortion correction
# gt: ground truth image without noise and artefacts 

baseurl="http://www.tractometer.org/downloads/downloads/"
dir="subjects/ismrm_$1"
mkdir -p "subjects"

# Basic ISMRM ##################################################################

if [ $1 = "basic" ]
then
    folder="ISMRM_2015_Tracto_challenge_data"

    curl "$baseurl${folder}_v1_1.zip" -o ${folder}_v1_1.zip

    unzip "${folder}_v1_1.zip"

    mv $folder $dir

    rm "${folder}_v1_1.zip"

    mv "${dir}/Diffusion.bvals" "${dir}/bvals"
    mv "${dir}/Diffusion.bvecs" "${dir}/bvecs"
    mv "${dir}/Diffusion.nii.gz" "${dir}/data.nii.gz"
fi

# Reverse Phase ISMRM ##########################################################

if [ $1 = "rpe" ]
then
    folder=ISMRM_2015_Tracto_challenge_data_with_reversed_phase

    curl "$baseurl${folder}_v1_0.zip" -o ${folder}_v1_0.zip

    unzip "${folder}_v1_0.zip"

    mv $folder $dir

    rm "${folder}_v1_0.zip"

    mv "${dir}/Diffusion_WITH_REVERSEPHASE.bvals" "${dir}/bvals"
    mv "${dir}/Diffusion_WITH_REVERSEPHASE.bvecs" "${dir}/bvecs"
    mv "${dir}/Diffusion_WITH_REVERSEPHASE.nii.gz" "${dir}/data.nii.gz"
fi

# Ground Truth ISMRM ###########################################################

if [ $1 = "gt" ]
then
    folder=ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2

    curl "${baseurl}ismrm_challenge_2015/${folder}.zip" -o ${folder}.zip

    unzip "${folder}.zip"

    mv $folder $dir

    rm "${folder}.zip"

    mv "${dir}/NoArtifacts_Relaxation.bvals" "${dir}/bvals"
    mv "${dir}/NoArtifacts_Relaxation.bvecs" "${dir}/bvecs"
    mv "${dir}/NoArtifacts_Relaxation.nii.gz" "${dir}/data.nii.gz"
fi

if [ ! -d "scoring/scoring_data" ]
then
    tar_file="scoring_data_tractography_challenge.tar.gz"
    mkdir "scoring"
    curl "http://www.tractometer.org/downloads/downloads/${tar_file}" \
    -o "scoring/${tar_file}"
    tar -xzf "scoring/${tar_file}" -C "scoring"
    rm "scoring/${tar_file}"
fi
