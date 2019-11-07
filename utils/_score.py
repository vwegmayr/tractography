import os
import sys
import subprocess
import argparse


def score_on_tm(fiber_path, blocking=True):

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

    cmd = []

    if "trimmed" not in fiber_path and "tm_all_merged" not in fiber_path:
        ismrm_version = fiber_path.split("/")[1].split("_")[1]
        trimmed_path = fiber_path[:-4] + "_{}_trimmed.trk".format(ismrm_version)
        cmd = [
            "track_vis", fiber_path, "-nr", "-l", "30", "200", "-o", trimmed_path, "&&"
        ]
    else:
        trimmed_path = fiber_path

    cmd += [
        "python", "scoring/scripts/score_tractogram.py", trimmed_path,
        "--base_dir", "scoring/scoring_data",
        "--out_dir", os.getcwd(),
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

    args = parser.parse_args()

    score_on_tm(args.trk_path)