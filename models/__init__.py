from .model_classes import FvM, FvMHybrid, RNNGRU, Entrack, RNNLSTM

MODELS = {"FvM": FvM,
          "FvMHybrid": FvMHybrid,
          "RNNGRU": RNNGRU,
          "Entrack": Entrack,
          'RNNLSTM': RNNLSTM}