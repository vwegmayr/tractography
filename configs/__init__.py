import os
import yaml

import models

import tensorflow.keras.backend as K
import numpy as np

from utils import Temperature


def compile_from(config_path, args, more_args):

    if config_path is not None:

        assert os.path.exists(config_path)

        with open(config_path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)

    else:
        config = {}

    args = {k: v for k,v in vars(args).items() if v is not None}

    config.update(args)

    for x in more_args:
        assert "--" in x[0]

    more_args = {x[0][2:]: x[1] for x in more_args if x[1] is not None}

    config.update(more_args)

    return config


def check(config):
    
    assert isinstance(config, dict)

    # ==========================================================================

    assert "model_name" in config

    assert "model_type" in config

    assert config["model_type"] in ["prior", "conditional"]

    assert config["model_name"] in list(models.MODELS.keys())

    assert "train_path" in config
    assert os.path.exists(config["train_path"])

    if "eval_path" in config:
        assert os.path.exists(config["eval_path"])

    assert "epochs" in config
    assert "batch_size" in config

    assert "optimizer" in config
    assert "optimizer_params" in config
    assert "learning_rate" in config["optimizer_params"]

    # ==========================================================================

    models.MODELS[config["model_name"]].check(config)

    return config


def save(config):

    sanitize(config)

    config_path = os.path.join(config["out_dir"], "config" + ".yml")
    print("\nSaving {}".format(config_path))
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def deep_update(config, update_dict):

    if isinstance(config, dict):
        config.update((k, v) for k, v in update_dict.items() if k in config)

        for v in config.values():
            deep_update(v, update_dict)


def sanitize(config):

    if isinstance(config, dict):
        
        for k, v in config.items():

            if isinstance(v, dict):
                sanitize(v)

            elif isinstance(v, Temperature):
                config[k] = float(np.round(K.get_value(v), 6))

            elif not (isinstance(v, (str, list)) or is_number(v)):
                config[k] = None



def is_number(obj):
    try:
        return (obj * 0) == 0
    except:
        return False