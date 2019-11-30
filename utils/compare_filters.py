from os.path import join, isdir
from os import listdir
import argparse
import json
import yaml
import itertools
import numpy as np
import matplotlib.pyplot as plt

SCORES = ['mean_F1', 'IC', 'IB', 'VC', 'VB', 'mean_OR', 'mean_OL']
BASELINES = [0.47369345142021646, 0.4257722385427191, 116, 0.5372817904136711,
             24, 0.34630964300920797, 0.45358382820115767]
FILTERS = ["log_prob_ratio", "log_prob_sum", "log_prob", "curvature"]


def compare(args):
    for i, score in enumerate(SCORES):
        baseline = BASELINES[i]
        compare_score(args, score, baseline)


def compare_score(args, score_name='mean_F1', baseline=0.47369345142021646):

    assert all(args.percentiles[i] <= args.percentiles[i+1]
               for i in range(len(args.percentiles)-1)), 'percentiles must be sorted'

    # Add one list for each filtering criteria
    criteria_scores = {'baseline': []}

    # Append values from json to each list
    if args.action == "track_vis":
        criteria_scores['track_vis'] = []
        for curv in args.max_curv:
            scoring_dir = join(args.results_path, f"scorings_trackvis_c-{curv}")
            if not isdir(scoring_dir):
                raise FileNotFoundError(f'File {scoring_dir} does not exist!')

            scoring_dir = join(scoring_dir, "scores")
            json_path = [file for file in listdir(scoring_dir)
                         if file.endswith('.json')][0]

            # Un comment for local use!
            # json_path = join(args.results_path, f'trackvis_{curv}.json')

            with open(join(scoring_dir, json_path)) as json_file:
                scores = json.load(json_file)

            criteria_scores['track_vis'].append(scores[score_name])
            criteria_scores['baseline'].append(baseline)

    else:
        for criteria in args.criteria:
            criteria_scores[criteria] = []

        for percentile in args.percentiles:
            for criteria in args.criteria:
                ext = 'bund' if args.action == 'bundle_filter' else 'fib'
                scoring_dir = join(args.results_path,
                                   f"scorings_p_{percentile}-f_{criteria}_{ext}")
                if not isdir(scoring_dir):
                    raise FileNotFoundError(f'File {scoring_dir} does not exist!')

                scoring_dir = join(scoring_dir, "scores")
                json_path = [file for file in listdir(scoring_dir)
                             if file.endswith('.json')][0]

                # Un comment for local use!
                # json_path = join(args.results_path, f'{criteria}_{percentile}_fib_k=f.json')
                # json_path = join(args.results_path, f'{criteria}_{percentile}_bund.json')

                with open(join(scoring_dir, json_path)) as json_file:
                    scores = json.load(json_file)

                criteria_scores[criteria].append(scores[score_name])
            criteria_scores['baseline'].append(baseline)

        # Info on bundles
        crit_pairs = list(set(itertools.product(args.criteria, args.criteria)))
        perc_pairs = list(itertools.product(args.criteria, args.criteria))
        for p1, p2 in perc_pairs:
            for c1, c2 in crit_pairs:
                assert c1 != c2

                removed_path_1 = join(args.result_path, f"removed_bundles_p-{p1}_f-{c1}")
                info_path_1 = join(removed_path_1, 'removed_info.yml')

                removed_path_2 = join(args.result_path, f"removed_bundles_p-{p2}_f-{c2}")
                info_path_2 = join(removed_path_2, 'removed_info.yml')

                with open(info_path_1, "r") as info_file:
                    info_file_1 = yaml.load(info_file)
                with open(info_path_2, "r") as info_file:
                    info_file_2 = yaml.load(info_file)

                bundles_1 = info_file_1['bundles']
                nb_bundles_removed_1 = info_file_1['nb_bundles_removed']
                nb_bundles_kept_1 = info_file_1['nb_bundles_kept']
                nb_bundles_1 = info_file_1['nb_bundles']
                bundles_removed_idx_1 = set(info_file_1['bundles_removed_idx'])
                bundles_kept_idx_1 = set(info_file_1['bundles_kept_idx'])
                avg_nb_fiber_1 = info_file_1['avg_nb_fiber']
                mean_avg_fiber_len_1 = info_file_1['mean_avg_fiber_len']
                median_avg_fiber_len_1 = info_file_1['median_avg_fiber_len']

                bundles_2 = info_file_2['bundles']
                nb_bundles_removed_2 = info_file_2['nb_bundles_removed']
                nb_bundles_kept_2 = info_file_2['nb_bundles_kept']
                nb_bundles_2 = info_file_2['nb_bundles']
                bundles_removed_idx_2 = set(info_file_2['bundles_removed_idx'])
                bundles_kept_idx_2 = set(info_file_2['bundles_kept_idx'])
                avg_nb_fiber_2 = info_file_2['avg_nb_fiber']
                mean_avg_fiber_len_2 = info_file_2['mean_avg_fiber_len']
                median_avg_fiber_len_2 = info_file_2['median_avg_fiber_len']

                assert nb_bundles_2 == nb_bundles_1
                both_removed = bundles_removed_idx_1 & bundles_removed_idx_2
                both_kept = bundles_kept_idx_1 & bundles_kept_idx_2
                onlyc1 = bundles_removed_idx_1 - bundles_removed_idx_2
                onlyc2 = bundles_removed_idx_2 - bundles_removed_idx_1

                c1_avg_fib_len = np.mean([b['avg_fib_len'] for b in bundles_1 if b['index'] in onlyc1]).item()
                c1_nb_fiber = np.mean([b['nb_fiber'] for b in bundles_1 if b['index'] in onlyc1]).item()

                c2_avg_fib_len = np.mean([b['avg_fib_len'] for b in bundles_2 if b['index'] in onlyc2]).item()
                c2_nb_fiber = np.mean([b['nb_fiber'] for b in bundles_2 if b['index'] in onlyc2]).item()

                comparison = {'xboth_removed': list(both_removed), 'xboth_kept': list(both_kept),
                              f'xonly_{c1}_removed': list(onlyc1),
                              f'xonly_{c2}_removed': list(onlyc2), f'only_{c1}_removed_avg_fib_len': c1_avg_fib_len,
                              f'only_{c2}_removed_avg_fib_len': c2_avg_fib_len,
                              f'only_{c1}_removed_nb_fiber': c1_nb_fiber,
                              f'only_{c2}_removed_nb_fiber': c2_nb_fiber, f'{c1}_avg_nb_fiber': avg_nb_fiber_1,
                              f'{c2}_avg_nb_fiber': avg_nb_fiber_2,
                              f'{c1}_mean_avg_fiber_len': mean_avg_fiber_len_1,
                              f'{c2}_mean_avg_fiber_len': mean_avg_fiber_len_2,
                              f'{c1}_median_avg_fiber_len': median_avg_fiber_len_1,
                              f'{c2}_median_avg_fiber_len': median_avg_fiber_len_2,
                              f'{c1}_nb_bundles_removed': nb_bundles_removed_1,
                              f'{c2}_nb_bundles_removed': nb_bundles_removed_2}

                comp_path = join(args.result_path, f'comp_{c1}_{c2}_p1-{p1}_p2-{p2}.yml')
                print(f'Saving comparison to {comp_path}...')
                with open(comp_path, "w") as file:
                    yaml.dump(comparison, file, default_flow_style=False)

    x_axis = args.max_curv if args.action == "track_vis" else args.percentiles
    fig, ax = plt.subplots()
    for key, values in criteria_scores.items():
        if len(values) > 0:
            if key == 'baseline':
                ax.plot(x_axis, values, label=key, linestyle='dashed')
            else:
                ax.plot(x_axis, values, label=key)
    legend = ax.legend()
    plt.title(f'{score_name}')
    fig_path = join(args.results_path, f'compare_{args.action}_{args.criteria}_{score_name}.png')
    print(f'Saving plot to {fig_path}')
    plt.savefig(fig_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Filter unlikely fibers.")
    parser.add_argument("--action", type=str, default='bundle_filter',
                        choices=['bundle_filter', 'fiber_filter', 'track_vis'])
    parser.add_argument("results_path", type=str)
    parser.add_argument('--percentiles', nargs='+', type=int, default=[],
                        help="list of percentiles to try")
    parser.add_argument('--criteria', nargs='+', type=str, default=FILTERS,
                        help="list of criteria to try")
    parser.add_argument('--max_curv', nargs='+', type=str, default=[],
                        help="list of criteria to try")
    args = parser.parse_args()

    compare(args)
