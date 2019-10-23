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

keep_frac=0.2
ext_num=020   # 100 * keep_frac - must be 3 digits integer
merge_out_dir="merged_tracts"

regex="hcp_zips/([0-9]+)_3T_Diffusion_preproc.zip"

for fileadress in "hcp_zips/"*".zip"; do
    if [[ "${fileadress}" =~ $regex ]]
    then
        filename="${BASH_REMATCH[1]}"
        
        echo "Unpacking ${filename} dwi files..." &&
        ./unpack_hcp_dwi.sh $filename &&
        
        echo "Unpacking ${filename} track files..." &&
        ./unpack_hcp_trks.sh $filename &&
        
        echo "Mergin ${filename} track files with keep ${keep_frac}..." &&
        python merge_tracks.py "subjects/${filename}/tracts" \
        --keep "${keep_frac}" \
        --weighted \
        --out_dir "subjects/${filename}/${merge_out_dir}" &&
        
        echo "Resampling ${filename} track files with keep ${keep_merge}..." &&
        python resample_trk.py "subjects/${filename}/${merge_out_dir}/merged_W${ext_num}.trk" &&
        
        echo "Estimating FOD for ${filename}..." &&
        ./est_fod ${filename}
        
    else
        echo "WARNING: ${fileadress} is not a HCP folder. Skipped from the entire pipeline."
    fi
done
