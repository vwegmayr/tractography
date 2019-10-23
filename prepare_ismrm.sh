#!/bin/bash

# Usage: ./prepare_ismrm.sh version
# where version is one of (basic, rpe, gt).
#
# Assumes nothin'
#
# Additional Requirements:
# MRtrix installation

echo "Downloading ismrm ${1} data and saving in subjects/ and scoring/scoring/data folders..."
./get_ismrm_data.sh $1

echo "preprocessing ismrm ${1} data..."
./preproc_ismrm.sh $1

echo "estimating FOD of ismrm ${1} data..."
./est_fod.sh "ismrm_${1}"
