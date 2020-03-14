import os
import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dipy.io.gradients import read_bvals_bvecs
import pickle
import multiprocessing

import yaml

from hashlib import md5
from configs import load

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)


def train_model(configs):

    hasher = md5()
    for v in configs.values():
        hasher.update(str(v).encode())

    out_dir = os.path.join(
        configs['out_dir'], configs["model_name"], hasher.hexdigest())
    if os.path.exists(out_dir):
        print("This model config has been trained already:\n{}".format(out_dir))
        return

    samples = np.load(configs["train_path"])
    inputs = samples["inputs"]
    outputs = samples["outgoing"]
    inputs = inputs[:min(configs.get("max_n_samples", np.inf), len(inputs))]
    outputs = outputs[:min(configs.get("max_n_samples", np.inf), len(outputs))]

    _, bvecs = read_bvals_bvecs(None, configs["bvecs"])

    output_classes = np.array([np.argmax([np.dot(base_vec, outvec)
                                          for base_vec in bvecs])
                               for outvec in outputs])

    clf = RandomForestClassifier(n_estimators=configs["n_estimators"],
                                 max_depth=configs["max_depth"],
                                 verbose=1,
                                 n_jobs=multiprocessing.cpu_count(),
                                 random_state=0)
    clf.fit(inputs, output_classes)

    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "config" + ".yml")
    print("Saving {}".format(config_path))
    with open(config_path, "w") as file:
        yaml.dump(configs, file, default_flow_style=False)

    model_path = os.path.join(out_dir, 'model')
    print("Saving {}".format(model_path))
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    return clf, inputs, outputs, output_classes


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train the entrack model")

    parser.add_argument("config_path", type=str, nargs="?",
                        help="Path to model config.")

    parser.add_argument("--max_n_samples", type=int,
                        help="Maximum number of samples to be used for both "
                             "training and evaluation")
    args = parser.parse_args()

    configs = load(args.config_path)
    if args.max_n_samples is not None:
        configs['max_n_samples'] = args.max_n_samples
    train_model(configs)
