from .model_classes import FvM, FvMHybrid, RNNGRU, Entrack, RNNLSTM, Detrack, \
    Trackifier

MODELS = {"FvM": FvM,
          "Detrack": Detrack,
          "FvMHybrid": FvMHybrid,
          "RNNGRU": RNNGRU,
          "Entrack": Entrack,
          'RNNLSTM': RNNLSTM,
          'Trackifier': Trackifier}