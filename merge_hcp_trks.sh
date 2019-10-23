#!/bin/bash

# Usage ./merge_hcp_trks.sh subjectID keepfrac
# Example: ./merge_hcp_trks.sh 917255 0.05

dir="subjects/${1}"

if [ ! -d "${dir}/tracts" ]
then
    echo "No tracts found, run unpack_hcp_trks.sh first!"
    exit
fi

python merge_tracks.py \
"${dir}/tracts" \
--keep "$2" \
--weighted