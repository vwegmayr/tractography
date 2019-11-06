import argparse
import glob
import yaml
from time import sleep

from pprint import pprint
from utils.config import load, make_configs_from
from train import train
from inference import run_inference
from multiprocessing import Process, SimpleQueue
from GPUtil import getAvailable

import sys, os

check_interval=5 # sec

   
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def priority_print(msg):
    enablePrint()
    print(msg)
    blockPrint()


def get_gpus():
    return getAvailable(limit=8, maxLoad=10**-6, maxMemory=10**-1)


def n_gpus():
    return len(get_gpus())


def model_path_glob_to_more_args(model_path):
    more_args = ["--model_path"]
    for path in glob.glob(model_path):
        more_args.append(path)
    return more_args


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dispatch several runs.")

    parser.add_argument("action", type=str, choices=["training", "inference"],
        default="training")

    parser.add_argument("base_config_path", type=str)

    args, more_args = parser.parse_known_args()

    config = load(args.base_config_path)

    assert config["action"] == args.action

    if args.action == "inference" and "*" in config["model_path"]:
        assert "--model_path" not in more_args
        more_args += model_path_glob_to_more_args(config["model_path"])

    configurations = make_configs_from(config, more_args)

    target = train if args.action == "training" else run_inference

    gpu_queue = SimpleQueue()
    for idx in get_gpus():
        gpu_queue.put(str(idx))

    procs = []

    try:
        blockPrint()
        while len(configurations) > 0:
            while not gpu_queue.empty() and len(configurations) > 0:
                p = Process(target=target, args=(configurations.pop(), gpu_queue))
                procs.append(p)
                p.start()
                sleep(3) # Wait to make sure the timestamp is different
            sleep(check_interval)

    except KeyboardInterrupt:
        for p in procs:
            p.join()
            while p.exitcode is None:
                sleep(0.1)