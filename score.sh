#!/bin/bash

# Usage: ./score.sh trkdir

trkfile="${1}/fibers.trk"

python "scoring/validate_tracts_space.py" $trkfile &&

python "scoring/scripts/score_tractogram.py" \
$trkfile \
--base_dir "scoring/scoring_data" \
--out_dir $1 \
--save_full_vc \
--save_full_ic \
--save_full_nc \
--save_ib \
--save_vb \
-v