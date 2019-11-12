import os
import sys
import subprocess
import argparse

from utils.trim import trim


def score(trk_path, out_dir=None, min_length=30, max_length=200, no_trim=False,
    blocking=True):

    env = os.environ.copy()
    if 'CONDA_PREFIX' in env:
        env_name = str(env["CONDA_DEFAULT_ENV"])
        env["CONDA_DEFAULT_ENV"] = "scoring"
        env["CONDA_PREFIX"] = env["CONDA_PREFIX"].replace(env_name, "scoring")
        env["PATH"] = env["PATH"].replace(
            os.path.join("envs", env_name), os.path.join("envs", "scoring")
            )
        env["_"] = env["_"].replace(
            os.path.join("envs", env_name), os.path.join("envs", "scoring")
            )

    if not no_trim:
        trk_path = trim(trk_path, min_length, max_length)

    if out_dir is None:
        out_dir = os.getcwd()

    cmd = [
        "python", "scoring/scripts/score_tractogram.py", trk_path,
        "--base_dir", "scoring/scoring_data",
        "--out_dir", out_dir,
        "--save_full_vc", "--save_full_ic", "--save_full_nc", "--save_ib",
        "--save_vb"
    ]

    cmd = " ".join(cmd)

    if sys.version_info >= (3, 5):
        if blocking:
            return subprocess.run(cmd, env=env, shell=True)
        else:
            return subprocess.Popen(cmd, env=env, shell=True)
    else:
        subprocess.call(cmd, shell=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Score trk file on Tractometer")

    parser.add_argument("trk_path", type=str)

    parser.add_argument("--no_trim", action="store_true")

    args = parser.parse_args()

    score(args.trk_path, no_trim=args.no_trim)