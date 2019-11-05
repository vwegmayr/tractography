import argparse
import yaml
from time import sleep

from pprint import pprint
from utils.config import load, make_configs_from
from train import train
from multiprocessing import Process
from GPUtil import getAvailable

import sys, os


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def priority_print(msg):
    enablePrint()
    print(msg)
    blockPrint()


def n_gpus():
    return len(getAvailable(limit=8, maxLoad=10**-6, maxMemory=10**-2))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dispatch several runs.")

    parser.add_argument("config_path", type=str)

    args, more_args = parser.parse_known_args()

    config = load(args.config_path)

    configurations = make_configs_from(config, more_args)

    proc_queue = [Process(target=train, args=(c, )) for c in configurations]
    proc_running = []

    try:
        blockPrint()
        while len(proc_queue) > 0:
            for p in proc_queue:
                if n_gpus() > 0:
                    p.start()
                    proc_running.append(proc_queue.pop(proc_queue.index(p)))
                    sleep(1.5)
            sleep(1)
    except KeyboardInterrupt:
        for p in proc_running:
            p.join()
            while p.exitcode is None:
                sleep(0.1)