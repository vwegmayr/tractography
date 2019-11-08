#!/bin/bash

# Usage: ./unpack_hcp_trks.sh subjectID
# Example: ./unpack_hcp_trks.sh 917255

trk_dir="HCP105_Zenodo_NewTrkFormat"
trk_url="https://zenodo.org/record/1477956/files/${trk_dir}.zip?download=1"

if [ ! -f "hcp_zips/hcp_trks.zip" ]
then
    echo "Downloading HCP trk files, this can take a while..."
    curl $hcp_trk_url -o "hcp_zips/hcp_trks.zip"
fi

if [ ! -d "subjects" ]
then
    mkdir "subjects"
fi

dir="subjects/${1}"

if [ ! -d "${dir}/tracts" ]
then
    unzip "hcp_zips/hcp_trks.zip" "${trk_dir}/${1}/*" -d "subjects"
    cp -r "subjects/${trk_dir}/${1}/tracts" "${dir}"
    rm -rf "subjects/${trk_dir}"
fi
