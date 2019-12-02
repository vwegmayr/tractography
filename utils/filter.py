from os.path import join, dirname
from os import makedirs
import argparse
from multiprocessing import Process
from subprocess import call

import nibabel as nib
import numpy as np
import yaml

from utils.config import load
from utils._score import score
from resample_trk import fiber_curvature
from nibabel.streamlines.trk import TrkFile
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.metric import ResampleFeature


FILTERS = ["log_prob_ratio", "log_prob_sum", "log_prob", "curvature", "none"]


def track_vis_filter(config, name='filter_run'):

    out_dir = join(dirname(config["trk_path"]))

    filtered_path = join(out_dir, f"trackvis_{config['max_curv']}.trk")

    command = f"track_vis {config['trk_path']} " \
              f"--curvature 0 {config['max_curv']} " \
              f"-l 30 200 -nr -o {filtered_path}"

    status = call(['/bin/bash', '-c', command])
    print(f"{name}: Saved {filtered_path}")

    if config["score"]:
        score(
            filtered_path,
            out_dir=join(out_dir, "scorings_{0}".format(name)),
            no_trim=True,
            python2=config['python2']
            )


def filter_fibers(config, name='filter_run'):

    trk_file = config['trk_file']
    tractogram = trk_file.tractogram

    if (config["filter_name"] != "none" and
        config["filter_name"] not in tractogram.data_per_point):
        raise ValueError("You need to mark fibers before filtering!")
    
    keep = list(range(len(tractogram)))

    if config["filter_name"] != "none":
        print("{0}: filter based on {1}...".format(name, config["filter_name"]))
        values = [min(t)[0] for t in tractogram.data_per_point[config["filter_name"]]]
        threshold_value = np.percentile(values, config["percentile"])
        print(f"Threshold value {threshold_value} with percentile {config['percentile']}")

        for i, tract in enumerate(tractogram.data_per_point[config["filter_name"]]):
            if min(tract)[0] < threshold_value:
                keep.remove(i)

    tractogram = tractogram[keep]

    out_dir = join(dirname(config["marked_trk_path"]))

    filtered_path = join(out_dir, "{}_{}_fib_k=f.trk".format(
        config["filter_name"], config["percentile"]))

    print("{0}: Saving {1}".format(name, filtered_path))
    TrkFile(tractogram, trk_file.header).save(filtered_path)

    if config["score"]:
        score(
            filtered_path,
            out_dir=join(out_dir, "scorings_{0}".format(name)),
            no_trim=True,
            python2=config['python2']
            )


def filter_bundles(config, name='filter_run'):
    out_dir = dirname(config["marked_trk_path"])
    removed_out_dir = join(out_dir, f"removed_bundles_p-{config['percentile']}_f-{config['filter_name']}")
    makedirs(removed_out_dir, exist_ok=True)

    trk_file = config['trk_file']
    bundles = config['bundles']
    tractogram = trk_file.tractogram

    if (config["filter_name"] != "none" and config['filter_name'] != 'curvature'
            and config["filter_name"] not in tractogram.data_per_point):
        raise ValueError("You need to mark fibers before filtering!")

    keep = list(range(len(tractogram)))

    print(f"{name}: {len(bundles.clusters)} clusters found")
    values = np.zeros(len(bundles.clusters))
    for i, b in enumerate(bundles.clusters):
        if config['filter_name'] != 'curvature':
            cluster_tracts = \
                tractogram.data_per_point[config["filter_name"]][b.indices]
            values[i] = np.mean([min(points)[0] for points in cluster_tracts])
        else:
            cluster_tracts = tractogram.streamlines[b.indices]
            curvatures = [fiber_curvature(fiber) for fiber in cluster_tracts]
            values[i] = 1 / np.mean([max(fiber)[0] for fiber in curvatures])

    threshold_value = np.percentile(values, config["percentile"])
    print(f"Threshold value {threshold_value} "
          f"with percentile {config['percentile']}")

    print(f"{name}: Filtering bundles ...")
    filtered_bundles = []
    filtered_bundles_idx = []
    kept_bundles_idx = []
    for i, cluster_value in enumerate(values):
        if cluster_value < threshold_value:

            # Save bundle info
            filtered_bundles_idx.append(i)
            this_bundle_file = join(removed_out_dir,
                                    f"removed_id-{i}_p-{config['percentile']}_f-{config['filter_name']}.trk")
            this_bundle = tractogram[bundles.clusters[i].indices]
            streamline_lengths = [len(s) for s in this_bundle.streamlines]
            this_bundle_info = {"index": i,
                                "nb_fiber": len(this_bundle),
                                "avg_fib_len": np.mean(streamline_lengths).item(),
                                "median_fib_len": np.median(streamline_lengths).item(),
                                'file': this_bundle_file}
            TrkFile(this_bundle, trk_file.header).save(this_bundle_file)
            filtered_bundles.append(this_bundle_info)

            # Finally remove the fibers in that bundle
            for index in bundles.clusters[i].indices:
                keep.remove(index)
        else:
            kept_bundles_idx.append(i)

    tractogram = tractogram[keep]
    print(f"{name}: {len(filtered_bundles)} bundles removed")

    nb_bundles_removed = len(filtered_bundles)
    nb_bundles_kept = len(kept_bundles_idx)
    nb_bundles = len(values)
    bundles_removed_idx = [b['index'] for b in filtered_bundles]
    bundles_kept_idx = kept_bundles_idx

    avg_nb_fiber = np.mean([b['nb_fiber'] for b in filtered_bundles]).item()
    mean_avg_fiber_len = np.mean([b['avg_fib_len'] for b in filtered_bundles]).item()
    median_avg_fiber_len = np.median([b['avg_fib_len'] for b in filtered_bundles]).item()
    filtered_bundles = {'bundles': filtered_bundles,
                        'nb_bundles_removed': nb_bundles_removed,
                        'nb_bundles_kept': nb_bundles_kept,
                        'nb_bundles': nb_bundles,
                        'bundles_removed_idx': bundles_removed_idx,
                        'bundles_kept_idx': bundles_kept_idx,
                        'avg_nb_fiber': avg_nb_fiber,
                        'mean_avg_fiber_len': mean_avg_fiber_len,
                        'median_avg_fiber_len': median_avg_fiber_len}
    with open(join(removed_out_dir, 'removed_info.yml'), "w") as file:
            yaml.dump(filtered_bundles, file, default_flow_style=False)
    print(f"{name}: average number of fibers: {avg_nb_fiber} | mean average length: {mean_avg_fiber_len}")

    filtered_path = join(out_dir, f"{config['filter_name']}_{config['percentile']}_bund.trk")

    print("{0}: Saving {1}".format(name, filtered_path))
    TrkFile(tractogram, trk_file.header).save(filtered_path)

    if config["score"]:
        score(
            filtered_path,
            out_dir=join(out_dir, "scorings_{0}".format(name)),
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

        print("Loading fibers ...")
        config['trk_file'] = nib.streamlines.load(config["marked_trk_path"])

        filter_func = None
        if args.action == 'fiber_filter':
            filter_func = filter_fibers
        elif args.action == 'bundle_filter':
            filter_func = filter_bundles

            print("Clustering fibers ...")
            feature = ResampleFeature(nb_points=config['centroid_size'])
            qb = QuickBundles(
                threshold=config['cluster_thresh'],
                metric=AveragePointwiseEuclideanMetric(feature)
            )

            bundles = qb.cluster(config['trk_file'].tractogram.streamlines)
            bundles.refdata = config['trk_file'].tractogram
            config['bundles'] = bundles

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
