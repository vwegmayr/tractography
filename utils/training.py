import os
import logging
import datetime

import tensorflow as tf
import numpy as np

from GPUtil import getFirstAvailable

from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.keras import backend as K

from utils import summaries as tracking_summaries
from utils import callbacks as tracking_callbacks
from tensorflow.python.keras import callbacks as keras_callbacks


class Temperature(ResourceVariable):
    """docstring for Temperature"""
    def __init__(self, T=0.0, name="Temperature"):
        super(Temperature, self).__init__(T, name=name)

    def get_config(self):
        return {"T": float(K.get_value(self))}


def setup_env(func):

    def setup_env_and_run(*args, **kwargs):

        os.environ['PYTHONHASHSEED'] = '0'
        tf.compat.v1.set_random_seed(3)
        np.random.seed(3)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        return func(*args, **kwargs)

    return setup_env_and_run


def maybe_get_a_gpu():
    return str(getFirstAvailable(
            order="load", maxLoad=10 ** -6, maxMemory=10 ** -1)[0])


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def parse_callbacks(config):
    callbacks = []

    for cb, kwargs in config.items():
        if hasattr(keras_callbacks, cb):
            CB = getattr(keras_callbacks, cb)(**kwargs)
        elif hasattr(tracking_summaries, cb):
            CB = getattr(tracking_summaries, cb)(**kwargs)
        elif hasattr(tracking_callbacks, cb):
            CB = getattr(tracking_callbacks, cb)(**kwargs)
        callbacks.append(CB)

    return callbacks