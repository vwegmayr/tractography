import argparse
import glob
import yaml
from time import sleep

from pprint import pprint
from utils.config import load, make_configs_from
from utils._score import score_on_tm
from utils._mark import mark
from train import train
from inference import run_inference
from multiprocessing import Process, SimpleQueue
from GPUtil import getAvailable

import sys, os

check_interval=5 # sec

ACTIONS = ["training", "inference", "scoring", "mark"]
   

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


def glob_to_more_args(glob_path, key):
    more_args = ["--" + key]
    for path in glob.glob(glob_path):
        more_args.append(path)
    return more_args


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dispatch several runs.")

    parser.add_argument("action", type=str, choices=ACTIONS,
        default="training")

    parser.add_argument("base_config_path", type=str)

    args, more_args = parser.parse_known_args()

    config = load(args.base_config_path)

    assert config["action"] == args.action

    procs = []

    if args.action in ["train", "inference"]:

        if args.action == "inference" and "*" in config["model_path"]:
            assert "--model_path" not in more_args
            more_args += glob_to_more_args(config["model_path"], "model_path")

        configurations = make_configs_from(config, more_args)

        target = train if args.action == "training" else run_inference

        gpu_queue = SimpleQueue()
        for idx in get_gpus():
            gpu_queue.put(str(idx))


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

    elif args.action == "scoring":

        for fiber_path in config["fiber_path"]:
            print("Scoring", fiber_path.split("/")[-1], "...")
            procs.append(score_on_tm(fiber_path, blocking=False))

        while any(p.poll() is None for p in procs):
            sleep(1)

    elif args.action == "mark":

        if "*" in config["model_path"]:
            assert "--model_path" not in more_args
            more_args += glob_to_more_args(config["model_path"], "model_path")

        if "*" in config["fiber_path"]:
            assert "--fiber_path" not in more_args
            more_args += glob_to_more_args(config["fiber_path"], "fiber_path")

        configurations = make_configs_from(config, more_args)

        gpu_queue = SimpleQueue()
        for idx in get_gpus():
            gpu_queue.put(str(idx))

        try:
            blockPrint()
            while len(configurations) > 0:
                while not gpu_queue.empty() and len(configurations) > 0:
                    p = Process(target=mark, args=(configurations.pop(), gpu_queue))
                    procs.append(p)
                    p.start()
                    sleep(3) # Wait to make sure the timestamp is different
                sleep(check_interval)

        except KeyboardInterrupt:
            for p in procs:
                p.join()
                while p.exitcode is None:
                    sleep(0.1)

    else:
        print("Invalid action {}, must be in {}".format(args.action, ACTIONS))