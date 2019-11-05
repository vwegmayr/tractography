import os
import yaml

from functools import reduce
import operator

from collections import Mapping
from itertools import chain, product
from operator import add

from copy import deepcopy

_FLAG_FIRST = object()

def flattenDict(d, join=add, lift=lambda x:x):
    results = []
    def visit(subdict, results, partialKey):
        for k,v in subdict.items():
            newKey = lift(k) if partialKey==_FLAG_FIRST else join(partialKey,lift(k))
            if isinstance(v,Mapping):
                visit(v, results, newKey)
            else:
                results.append((newKey,v))
    visit(d, results, _FLAG_FIRST)
    return results


def load(config_path, attr=None):
    with open(config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    return config if attr is None else config[attr]


def group_more_args(args):

    nargs = len(args)
    grouped_args = []
    i=0
    while i < nargs:
        if "--" in args[i]:
            grouped_args.append([args[i][2:], []])
            i+=1
            while (i < nargs) and ("--" not in args[i]):
                grouped_args[-1][1].append(args[i])
                i+=1

    return grouped_args


def grouped_more_args_to_updates(grouped_args, multiple_values):

    updates = {}
    for a in grouped_args:
        keys = a[0].split(".")
        keys.reverse()
        if a[1][0].isnumeric():
            val = [int(v) for v in a[1]]
        elif a[1][0].replace(".", "", 1).isnumeric():
            val = [float(v) for v in a[1]]
        else:
            val = [v for v in a[1]]
        if multiple_values:
            d = {keys[0]: val}
        else:
            d = {keys[0]: val[0]}
        for k in keys[1:]:
            d = {k: d}

        nested_update(updates, d, replace_none=True)
        print()
        #updates.update(d)

    return updates


def parse_more_args(more_args, multiple_values=False):

    g = group_more_args(more_args)

    return grouped_more_args_to_updates(g, multiple_values)


def deep_update(config, update_dict):
    """Replaces all values at any level, if keys match."""
    if isinstance(config, dict):
        config.update((k, v) for k, v in update_dict.items() if k in config)

        for v in config.values():
            deep_update(v, update_dict)


def nested_update(config, update_dict, replace_none=False):
    """"""
    for k, v in update_dict.items():
        if isinstance(v, dict):
            if k in config:
                nested_update(config[k], v, replace_none)
            elif replace_none:
                config[k] = {}
                nested_update(config[k], v, replace_none)
        elif (k in config and config[k] is not None) or replace_none:
                config[k] = v


def sanitize(config):

    if isinstance(config, dict):
        
        for k, v in config.items():

            if isinstance(v, dict):
                sanitize(v)

            elif hasattr(v, "numpy"):
                config[k] = float(v.numpy())

            elif not (isinstance(v, (str, list)) or is_number(v)):
                config[k] = None


def is_number(obj):
    try:
        return (obj * 0) == 0
    except:
        return False


def get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)


def set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    get_by_path(root, items[:-1])[items[-1]] = value


def make_configs_from(base_config, more_args):

    changes = parse_more_args(more_args, multiple_values=True)
    flat_changes = flattenDict(changes, lift=lambda x:[x,])

    configs=[]
    for i, change in enumerate(flat_changes):
        for v in change[1]:
            d = deepcopy(base_config)
            set_by_path(d, change[0], v)
            configs.append(d)

    all_changes = [[[c[0], v] for v in c[1]] for c in flat_changes]

    configs = []
    for c in product(*all_changes):
        d = deepcopy(base_config)
        for v in c:
            set_by_path(d, v[0], v[1])
        configs.append(d)

    return configs