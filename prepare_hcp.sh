#!/bin/bash

# Usage: ./prepare_hcp.sh
# Example: ./prepare_hcp.sh
#
# Assumes you manually downloaded one or more files like
# "917255_3T_Diffusion_preproc.zip" from
# https://db.humanconnectome.org to a folder named "hcp_zips".
# Accessing this data online requires only free registration.
#
# Additional Requirements:
# MRtrix installation

keep_merge=0.2
regex="hcp_zips/([0-9]+)_3T_Diffusion_preproc.zip"

for fileadress in "hcp_zips/"*".zip"; do
    if [[ "${fileadress}" =~ $regex ]]
    then
        filename="${BASH_REMATCH[1]}"
        echo "Unpacking ${filename} dwi files..." &&
        ./unpack_hcp_dwi.sh $filename &&
        echo "Unpacking ${filename} track files..." &&
        ./unpack_hcp_trks.sh $filename &&
        echo "Mergin ${filename} track files with keep ${keep_merge}..." &&
        ./merge_hcp_trks.sh $filename $keep_merge
    else
        echo "WARNING: ${fileadress} is not a HCP folder. Skipped."
    fi
done
