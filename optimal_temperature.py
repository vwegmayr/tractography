import os
import glob
import argparse

from time import sleep
from copy import deepcopy 
from multiprocessing import SimpleQueue, Process, Manager

from agreement import agreement
from inference import run_inference
from utils._dispatch import get_gpus
from utils.config import load
from utils.training import timestamp
from configs import save


def find_optimal_temperature(config):

    model_paths = glob.glob(config["model_glob"])

    dwi_path_1 = config["inference"]["dwi_path"].format("")
    dwi_path_2 = config["inference"]["dwi_path"].format("retest")

    gpu_queue = SimpleQueue()
    for idx in get_gpus():
        gpu_queue.put(str(idx))

    procs=[]
    pred_manager = Manager()
    predictions = pred_manager.dict()
    try:
        for mp in model_paths:

            #if any(t in mp for t in []):

            model_config = config["inference"].copy()
            model_config["model_path"] = mp

            for j in [0,1]:
                run_config = model_config.copy()
                parse(run_config, "dwi_path", j)
                parse(run_config, "prior_path", j)
                parse(run_config, "term_path", j)
                parse(run_config, "seed_path", j)
                while gpu_queue.empty():
                    sleep(10)

                p = Process(
                    target=run_inference,
                    args=(run_config, gpu_queue, predictions)
                )
                p.start()
                procs.append(p)
                print("Launched {}: {}".format(mp.split("/")[-1], j))
                sleep(10)

    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.join()
            while p.exitcode is None:
                sleep(0.1)

    pred_pairs = group_by_model(predictions)
    config["pred_pairs"] = pred_pairs

    save(config,
        name="opT_{}.yml".format(timestamp()),
        out_dir=os.path.dirname(config["model_glob"])
    )
    """
    gpu_queue = SimpleQueue()
    for idx in get_gpus()[:4]:
        gpu_queue.put(str(idx))

    print("\nLaunching Agreement")
    try:
        procs=[]
        for model_path, pair in pred_pairs.items():
            while gpu_queue.empty():
                sleep(10)

            p = Process(
                target=agreement,
                args=(model_path,
                      pair[0]["dwi_path"],
                      pair[0]["trk_path"],
                      pair[1]["dwi_path"],
                      pair[1]["trk_path"],
                      config["agreement"]["wm_path"],
                      config["agreement"]["fixel_cnt_path"],
                      config["agreement"]["cluster_thresh"],
                      config["agreement"]["centroid_size"],
                      config["agreement"]["fixel_thresh"],
                      config["agreement"]["bundle_min_cnt"],
                      gpu_queue)
            )
            procs.append(p)
            p.start()
            sleep(10)

    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.join()
            while p.exitcode is None:
                sleep(0.1)
    """

def parse(config, path, instance):
    config[path] = config[path].format("retest" if instance == 1 else "")


def group_by_model(predictions):

    model_dict = {}

    for trk_path, config in predictions.items():

        model_path = config["model_path"]
        dwi_path = config["dwi_path"]

        if model_path in model_dict:
            model_dict[model_path].append(
                {"trk_path": trk_path, "dwi_path": dwi_path}
            )
        else:
            model_dict[model_path] = [
                {"trk_path": trk_path, "dwi_path": dwi_path}
            ]

    return model_dict

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Calculate optimal temperature.")

    parser.add_argument("config_path", type=str)

    args = parser.parse_args()

    config = load(args.config_path)

    find_optimal_temperature(config)