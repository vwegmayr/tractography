import os

from utils.config import load
from tensorflow.keras.models import load_model as keras_load_model

from .model_classes import FvM, FvMHybrid, RNNGRU, Entrack, RNNLSTM, Detrack, \
    Trackifier, RNNGRUEntrack, RNNLSTMEntrack

MODELS = {"FvM": FvM,
          "Detrack": Detrack,
          "FvMHybrid": FvMHybrid,
          "RNNGRU": RNNGRU,
          "Entrack": Entrack,
          'RNNLSTM': RNNLSTM,
          'Trackifier': Trackifier,
          'RNNGRUEntrack': RNNGRUEntrack,
          'RNNLSTMEntrack': RNNLSTMEntrack}


def load_model(model_path):

    model_config_path = os.path.join(
        os.path.dirname(model_path), "config.yml")

    model_name = load(model_config_path, "model_name")

    if hasattr(MODELS[model_name], "custom_objects"):
        return keras_load_model(model_path,
                           custom_objects=MODELS[model_name].custom_objects,
                           compile=False)
    else:
        return keras_load_model(model_path, compile=False)