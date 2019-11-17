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
from configs import save

def find_optimal_temperature(config):

    model_paths = list(glob.glob(config["model_glob"]))

    dwi_path_1 = config["inference"]["dwi_path"].format("", "")
    dwi_path_2 = config["inference"]["dwi_path"].format("retest", "_aligned")

    gpu_queue = SimpleQueue()
    for idx in get_gpus():
        gpu_queue.put(str(idx))

    procs=[]
    pred_manager = Manager()
    predictions = pred_manager.dict()
    try:
        for mp in model_paths:

            model_config = config["inference"].copy()
            model_config["model_path"] = mp

            for j in [0,1]:
                run_config = model_config.copy()
                parse(run_config, "dwi_path", j)
                parse(run_config, "prior_path", j)
                parse(run_config, "term_path", j)
                parse(run_config, "seed_path", j)
                while gpu_queue.empty():
                    sleep(2)
                p = Process(
                    target=run_inference,
                    args=(run_config, gpu_queue, predictions)
                )
                p.start()
                procs.append(p)
                print("Launched {}: {}".format(mp.split("/")[-1], j))
                sleep(2)
    except KeyboardInterrupt:
        for p in procs:
            p.join()
            while p.exitcode is None:
                sleep(0.1)

    while len(predictions) < 2 * len(model_paths):
        sleep(1)

    pred_pairs = group_by_model(predictions)

    agreement_config = {}
    agreement_config["pred_pairs"] = deepcopy(pred_pairs)
    agreement_config["wm_path"] = config["wm_path"]
    agreement_config["matching_thresh"] = config["matching_thresh"]
    agreement_config["cluster_thresh"] = config["cluster_thresh"]
    agreement_config["centroid_size"] = config["centroid_size"]

    save(agreement_config,
        name="agreement_config.yml",
        out_dir=os.path.dirname(config["model_glob"])
    )

    print("\nLaunching Agreement")
    try:
        procs=[]
        for model_path, pair in pred_pairs.items():
            while gpu_queue.empty():
                sleep(2)
            p = Process(
                target=agreement,
                args=(model_path,
                      pair[0]["dwi_path"],
                      pair[0]["trk_path"],
                      pair[1]["dwi_path"],
                      pair[1]["trk_path"],
                      config["wm_path"],
                      config["matching_thresh"],
                      config["cluster_thresh"],
                      config["centroid_size"],
                      gpu_queue)
            )
            procs.append(p)
            p.start()
            sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.join()
            while p.exitcode is None:
                sleep(0.1)


def parse(config, path, instance):
    config[path] = config[path].format(
        "retest" if instance == 1 else "",
        "_aligned" if instance == 1 else ""
    )


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