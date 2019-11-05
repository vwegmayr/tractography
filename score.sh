#!/bin/bash

# Usage: ./score.sh trk_path prefix

out_dir="$(dirname $1)"

python "scoring/validate_tracts_space.py" $1 &&

mv $1 "${2}_${1}"

python "scoring/scripts/score_tractogram.py" \
"${2}_${1}" \
--base_dir "scoring/scoring_data" \
--out_dir $out_dir \
--save_full_vc \
--save_full_ic \
--save_full_nc \
--save_ib \
--save_vb \
-v

mv "${2}_${1}" $1