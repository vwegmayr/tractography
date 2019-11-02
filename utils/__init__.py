from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.keras import backend as K

class Temperature(ResourceVariable):
    """docstring for Temperature"""
    def __init__(self, T=0.0, name="Temperature"):
        super(Temperature, self).__init__(T, name=name)

    def get_config(self):
        return {"T": float(K.get_value(self))}