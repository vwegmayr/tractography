import os
import argparse
from multiprocessing import Process
from subprocess import call

import nibabel as nib
import numpy as np

from utils.config import load
from utils._score import score
from nibabel.streamlines.trk import TrkFile
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.metric import ResampleFeature


FILTERS = ["log_prob_ratio", "log_prob_sum", "log_prob", "none"]


def track_vis_filter(config, name='filter_run'):

    out_dir = os.path.join(os.path.dirname(config["trk_path"]))

    filtered_path = os.path.join(out_dir, f"trackvis_{config['max_curv']}.trk")

    command = f"track_vis {config['trk_path']} " \
              f"--curvature 0 {config['max_curv']} " \
              f"-l 30 200 -nr -o {filtered_path}"

    status = call(['/bin/bash', '-c', command])
    print(f"{name}: Saved {filtered_path}")

    if config["score"]:
        score(
            filtered_path,
            out_dir=os.path.join(out_dir, "scorings_{0}".format(name)),
            no_trim=True,
            python2=config['python2']
            )


def filter_fibers(config, name='filter_run'):

    print("{0}: Loading fibers ...".format(name))
    trk_file = nib.streamlines.load(config["marked_trk_path"])

    tractogram = trk_file.tractogram

    if (config["filter_name"] != "none" and
        config["filter_name"] not in tractogram.data_per_point):
        raise ValueError("You need to mark fibers before filtering!")
    
    keep = list(range(len(tractogram)))

    if config["filter_name"] != "none":
        print("{0}: filter based on {1}...".format(name, config["filter_name"]))
        values = [min(t)[0] for t in tractogram.data_per_point[config["filter_name"]]]
        threshold_value = np.percentile(values, config["percentile"])

        for i, tract in enumerate(tractogram.data_per_point[config["filter_name"]]):
            if min(tract)[0] < threshold_value:
                keep.remove(i)

    if "apply_k" in config and config["apply_k"]:
        print("{0}: filter based on curvatures...".format(name))
        curvatures = [t[0,0] for t in tractogram.data_per_point["k"]]
        k_threshold = np.percentile(curvatures, config["percentile"])
        for i, tract in enumerate(tractogram.data_per_point["k"]):
            if tract[0, 0] > k_threshold:
                try:
                    keep.remove(i)
                except ValueError:  # already removed before
                    pass

    tractogram = tractogram[keep]

    out_dir = os.path.join(os.path.dirname(config["marked_trk_path"]))

    filtered_path = os.path.join(out_dir, "{}_{}_fib_k={}.trk".format(
        config["filter_name"], config["percentile"],
        "t" if config["apply_k"] else "f"))

    print("{0}: Saving {1}".format(name, filtered_path))
    TrkFile(tractogram, trk_file.header).save(filtered_path)

    if config["score"]:
        score(
            filtered_path,
            out_dir=os.path.join(out_dir, "scorings_{0}".format(name)),
            no_trim=True,
            python2=config['python2']
            )


def filter_bundles(config, name='filter_run'):
    print("{0}: Loading fibers ...".format(name))
    trk_file = nib.streamlines.load(config["marked_trk_path"])

    tractogram = trk_file.tractogram

    if (config["filter_name"] != "none" and
            config["filter_name"] not in tractogram.data_per_point):
        raise ValueError("You need to mark fibers before filtering!")

    keep = list(range(len(tractogram)))

    print(f"{name}: Clustering fibers ...")
    feature = ResampleFeature(nb_points=config['centroid_size'])
    qb = QuickBundles(
        threshold=config['cluster_thresh'],
        metric=AveragePointwiseEuclideanMetric(feature)
    )

    bundles = qb.cluster(tractogram.streamlines)
    bundles.refdata = tractogram

    print(f"{name}: {len(bundles.clusters)} clusters found")
    values = np.zeros(len(bundles.clusters))
    for i, b in enumerate(bundles.clusters):
        cluster_tracts = tractogram.data_per_point[config["filter_name"]][b.indices]
        values[i] = np.mean([min(points)[0] for points in cluster_tracts])

    threshold_value = np.percentile(values, config["percentile"])

    print(f"{name}: Filtering bundles ...")
    filtered_bundles = []
    for i, cluster_value in enumerate(values):
        if cluster_value < threshold_value:
            filtered_bundles.append(i)
            for index in bundles.clusters[i].indices:
                keep.remove(index)
    tractogram = tractogram[keep]
    print(f"{name}: {len(filtered_bundles)} bundles removed: {filtered_bundles}")

    out_dir = os.path.join(os.path.dirname(config["marked_trk_path"]))

    filtered_path = os.path.join(out_dir, "{}_{}_bund.trk".format(
        config["filter_name"], config["percentile"]))

    print("{0}: Saving {1}".format(name, filtered_path))
    TrkFile(tractogram, trk_file.header).save(filtered_path)

    if config["score"]:
        score(
            filtered_path,
            out_dir=os.path.join(out_dir, "scorings_{0}".format(name)),
            no_trim=True,
            python2=config['python2']
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Filter unlikely fibers.")

    parser.add_argument("config_path", type=str)

    parser.add_argument("--action", type=str,
                        choices=['bundle_filter', 'fiber_filter', 'track_vis'])

    parser.add_argument('--percentiles', nargs='+', type=int,
                        help="list of percentiles to try")

    parser.add_argument('--max_curv', nargs='+', type=int,
                        help="list of maximum curvature for track_vis algorithm"
                             " to try")

    parser.add_argument('--criteria', nargs='+', type=str,
                        help="list of criteria to try")

    args = parser.parse_args()

    config = load(args.config_path)

    if args.action == 'track_vis':
        if args.max_curv is None:
            args.max_curv = [config['max_curv']]

        for curv in args.max_curv:
            print("Filtering with max curv {0}".format(curv))

            config["max_curv"] = curv

            name = 'trackvis_c-{0}'.format(curv)
            p = Process(target=track_vis_filter, args=(config, name))
            p.start()
    else:
        if args.percentiles is None:
            args.percentiles = [config['percentile']]

        if args.criteria is None:
            args.criteria = [config['filter_name']]

        filter_func = None
        if args.action == 'fiber_filter':
            filter_func = filter_fibers
        elif args.action == 'bundle_filter':
            filter_func = filter_bundles

        for criteria in args.criteria:
            print("Filtering with criteria {0}".format(criteria))
            for percentile in args.percentiles:
                print("Filtering with percentile {0}".format(percentile))

                config["percentile"] = percentile
                config["filter_name"] = criteria

                assert config["filter_name"] in FILTERS

                name = 'p_{0}-f_{1}_{2}'.format(
                    percentile, criteria,
                    'bund' if args.action == 'bundle_filter' else 'fib')
                p = Process(target=filter_func, args=(config, name))
                p.start()
