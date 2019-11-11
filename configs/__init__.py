import os
import yaml

from models import MODELS

from tensorflow.keras.backend import get_value

from utils.training import Temperature

from utils.config import *

from pprint import pprint

from utils.filelock import filelock


def compile_from(config_path, args, more_args):

    config = load(config_path)

    args = {k: v for k,v in vars(args).items() if v is not None}

    config.update(args)

    nested_update(config, parse_more_args(more_args))

    return config


def check(config):
    
    assert isinstance(config, dict)

    # ==========================================================================

    assert "model_name" in config

    if "model_type" in config:
        assert config["model_type"] in ["prior", "conditional"]

    assert config["model_name"] in list(MODELS.keys())

    assert "train_path" in config

    if isinstance(config["train_path"], list):
        assert all(os.path.exists(path) for path in config["train_path"])
    else:
        assert os.path.exists(config["train_path"])

    if "eval_path" in config:
        assert os.path.exists(config["eval_path"])

    assert "epochs" in config
    assert "batch_size" in config

    assert "optimizer" in config
    assert "opt_params" in config
    assert "learning_rate" in config["opt_params"]

    # ==========================================================================

    MODELS[config["model_name"]].check(config)

    return config


def save(config):

    sanitize(config)

    config_path = os.path.join(config["out_dir"], "config.yml")
    print("\nSaving {}".format(config_path))
    with filelock.FileLock("config_path"):   
        with open(config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)


def add(config, to=".running"):
    config_path = os.path.join(config["out_dir"], "config.yml\n")

    with filelock.FileLock(to): 
        with open(to, "a") as file:
            file.write(config_path)


def remove(config, _from=".running"):
    config_path = os.path.join(config["out_dir"], "config.yml")

    with filelock.FileLock(_from):

        with open(_from, "r") as file:
            runs = list(file.readlines())

        with open(_from, "w") as file:
            for run in runs:
                if run.strip("\n") != config_path:
                    file.write(run)

        if len(runs) == 1:
            os.remove(_from)