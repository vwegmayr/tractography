#!/bin/bash

# Usage: ./score.sh trkdir

trkfile="${1}/fibers.trk"
alignedfile="${1}/fibers_tm.trk"

python "../tractconverter/scripts/TractConverter.py" \
-i  $trkfile \
-o  $alignedfile \
-a "scoring_data/masks/wm2mm.nii.gz" &&

python "validate_tracts_space.py" $alignedfile &&

python "scripts/score_tractogram.py" \
$alignedfile \
"scoring_data" \
$1 \
--save_full_vc \
--save_full_ic \
--save_full_nc \
--save_ib \
--save_vb \
-v