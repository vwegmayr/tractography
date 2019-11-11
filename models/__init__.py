from .model_classes import FvM, FvMHybrid, RNNGRU, Entrack, RNNLSTM, Detrack

MODELS = {"FvM": FvM,
          "Detrack": Detrack,
          "FvMHybrid": FvMHybrid,
          "RNNGRU": RNNGRU,
          "Entrack": Entrack,
          'RNNLSTM': RNNLSTM}