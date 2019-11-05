import yaml
import os
import argparse
import sys
from pprint import pprint
from pandas import DataFrame, option_context
from flatten_dict import flatten
from copy import deepcopy

IGNORE = ["out_dir"]


def reduce_paths(path_list):
    return ["/".join([p.split("/")[i] for i in [1,3]]) for p in path_list]


def stripper(data):
    new_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = stripper(v)
        if not v in (u'', None, {}):
            new_data[k] = v
    return new_data



def dict_diff(dict_list, diff={}):

    common_keys = set.intersection(*[set(d.keys()) for d in dict_list])

    for k in common_keys:
        if k not in IGNORE:
            values = [d[k] for d in dict_list]
            if all([isinstance(v, dict) for v in values]):
                subdiff = {}
                diff[k] = subdiff
                dict_diff(values, subdiff)
            elif not all(v == values[0] for v in values) and k not in IGNORE:
                diff[k] = values

    diff = stripper(diff)

    return flatten(diff)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compare runs.")

    parser.add_argument("last_n", nargs="?", default=5, type=int)

    parser.add_argument("-a", action="store_true", dest="active")

    parser.add_argument("-f", action="store_true", dest="full")

    args = parser.parse_args()


    file_path = ".running" if args.active else ".archive"

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            runs = [r.strip("\n") for r in file.readlines()]
    else:
        print("No file {} found.".format(file_path))
        sys.exit()

    if len(runs) == 0:
        print("No runs found in {}.".format(file_path))
        sys.exit()
    else:
        runs = runs[-args.last_n:]
        configs = []
        for r in runs:
            if os.path.exists(r):
                with open(r, "r") as config_file:
                    configs.append(yaml.load(config_file,
                        Loader=yaml.FullLoader))

        time_stamps = {"timestamp": [r.split("/")[-2] for r in runs]}

        diffs = dict_diff(configs)

        new_diffs = {}
        for k,v in diffs.items():
            if args.full:
                new_diffs[".".join(k)] = diffs[k]
            elif isinstance(k, tuple):
                if "path" not in k[-1]:
                    new_diffs[k[-1]] = diffs[k]
                else:
                    new_diffs[k[-1]] = reduce_paths(diffs[k])


        time_stamps.update(new_diffs)

        with option_context('display.colheader_justify','left'):
            print(DataFrame(time_stamps).to_string(index=False))