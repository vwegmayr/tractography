import os
import argparse
from multiprocessing import Process

import nibabel as nib
import numpy as np

from utils.config import load
from utils._score import score
from nibabel.streamlines.trk import TrkFile


FILTERS = ["log_prob_ratio", "log_prob_sum", "log_prob", "none"]


def filter_fibers(config):

    print("Loading fibers ...")
    trk_file = nib.streamlines.load(config["trk_path"])

    tractogram = trk_file.tractogram

    if (config["filter_name"] != "none" and
        config["filter_name"] not in tractogram.data_per_point):
        raise ValueError("You need to mark fibers before filtering!")
    
    keep = list(range(len(tractogram)))

    if config["filter_name"] != "none":
        print("filter based on {0}...".format(config["filter_name"]))
        values = [t[0,0] for t in tractogram.data_per_point[config["filter_name"]]]
        threshold_value = np.percentile(values, config["percentile"])

        for i, tract in enumerate(tractogram.data_per_point[config["filter_name"]]):
            if tract[0, 0] < threshold_value:
                keep.remove(i)

    if "apply_k" in config and config["apply_k"]:
        print("filter based on curvatures...")
        curvatures = [t[0,0] for t in tractogram.data_per_point["k"]]
        k_threshold = np.percentile(curvatures, config["percentile"])
        for i, tract in enumerate(tractogram.data_per_point["k"]):
            if tract[0, 0] > k_threshold:
                try:
                    keep.remove(i)
                except ValueError:  # already removed before
                    pass

    tractogram = tractogram[keep]

    out_dir = os.path.dirname(config["trk_path"])

    filtered_path = os.path.join(out_dir, "{}_{}_k={}.trk".format(
        config["filter_name"], config["percentile"],
        "t" if config["apply_k"] else "f"))

    print("Saving {}".format(filtered_path))
    TrkFile(tractogram, trk_file.header).save(filtered_path)

    if config["score"]:
        score(
            filtered_path,
            out_dir=os.path.join(out_dir, "scorings"),
            no_trim=True,
            python2=config['python2']
            )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Filter unlikely fibers.")

    parser.add_argument("config_path", type=str)

    parser.add_argument('--percentiles', nargs='+', type=int,
                        help="list of percentiles to try")

    parser.add_argument('--criteria', nargs='+', type=str,
                        help="list of criteria to try")

    args = parser.parse_args()

    config = load(args.config_path)

    if args.percentiles is None:
        args.percentiles = [config['percentile']]

    if args.criteria is None:
        args.criteria = [config['filter_name']]

    for criteria in args.criteria:
        print("Filtering with criteria {0}".format(criteria))
        for percentile in args.percentiles:
            print("Filtering with percentile {0}".format(percentile))

            config["percentile"] = percentile
            config["filter_name"] = criteria

            assert config["filter_name"] in FILTERS

            p = Process(target=filter_fibers, args=config)
