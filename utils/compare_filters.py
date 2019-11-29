from os.path import join, isdir
from os import listdir
import argparse
import json

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
    fig_path = join(args.results_path, f'compare_{args.action}_filter_{score_name}.png')
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
