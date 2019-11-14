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

keep_frac=0.05
ext_num=0050   # 100 * keep_frac - must be 3 digits integer
merge_out_dir="merged_tracts"

regex="hcp_zips/([0-9]+[retest]*)_3T_Diffusion_preproc.zip"

if [ -z $1 ]
then
    paths="hcp_zips/"*".zip"
else
    paths="hcp_zips/${1}_3T_Diffusion_preproc.zip"
fi

for fileadress in $paths; do
    if [[ "${fileadress}" =~ $regex ]]
    then
        filename="${BASH_REMATCH[1]}"
        subjectID=$(grep -Po "[0-9]+" <<< ${filename})

        echo "Unpacking ${filename} dwi files..." &&
        ./unpack_hcp_dwi.sh $filename

        if [ ${filename} == ${subjectID} ] # Only HCP subjects have fibers
        then
            echo "Unpacking ${filename} track files..." &&
            ./unpack_hcp_trks.sh $filename &&

            echo "Merging ${filename} track files with keep ${keep_frac}..." &&
            python merge_tracks.py "subjects/${filename}/tracts" \
            --keep "${keep_frac}" \
            --weighted \
            --out_dir "subjects/${filename}/${merge_out_dir}" &&

            echo "Resampling ${filename} track files with keep ${keep_merge}..." &&
            python resample_trk.py "subjects/${filename}/${merge_out_dir}/merged_W${ext_num}.trk"
        fi

        echo "Estimating FOD for ${filename}..." &&
        ./est_fod.sh ${filename}
        
    else
        echo "WARNING: ${fileadress} is not a HCP folder. Skipped from the entire pipeline."
    fi
done
