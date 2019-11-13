import os
import sys
import subprocess
import argparse

if sys.version_info >= (3, 5):
    from utils.trim import trim
else:
    from trim import trim


def score(trk_path, out_dir=None, min_length=30, max_length=200, no_trim=False,
    blocking=True, python2=None):

    if not no_trim:
        trk_path = trim(trk_path, min_length, max_length)

    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "scoring/scripts/score_tractogram.py", trk_path,
        "--base_dir", "scoring/scoring_data",
        "--out_dir", out_dir,
        "--save_full_vc", "--save_full_ic", "--save_full_nc", "--save_ib",
        "--save_vb"
    ]

    cmd = " ".join(cmd)

    if python2:
        source_cmd = "source '{}' && " if '/' in python2 \
            else "source activate '{}'".format(python2)

        cmd = source_cmd + cmd

    if blocking:
        out = subprocess.run(['/bin/bash', '-c', cmd])
    else:
        out = subprocess.Popen(['/bin/bash', '-c', cmd])

    print(out.stdin)
    print(out.stderr)
    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Score trk file on Tractometer")

    parser.add_argument("trk_path", type=str)

    parser.add_argument("--no_trim", action="store_true")

    parser.add_argument("--python2", type=str)

    args = parser.parse_args()

    score(args.trk_path, no_trim=args.no_trim, python2=args.python2)
