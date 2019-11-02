import os

import numpy as np

from tensorflow.python.keras.callbacks import Callback
from tensorflow.keras import backend as K

from . import Temperature


class RNNResetCallBack(Callback):
    def __init__(self, reset_batches):
        super(RNNResetCallBack, self).__init__()
        self.reset_batches = reset_batches

    def on_batch_end(self, batch, logs={}):
        if batch in self.reset_batches:
            self.model.reset_states()
        return


class ConstantTemperatureSchedule(Callback):
    """docstring for ConstantTemperatureSchedule"""
    def __init__(self, T, *args, **kwargs):
        super(ConstantTemperatureSchedule, self).__init__(*args, **kwargs)
        self.T = T

    def schedule(self, step, logs={}):
        return float(K.get_value(self.T))

    def on_train_batch_end(self, batch, logs={}):
        t = self.schedule(batch, logs)
        K.set_value(self.T, t)
        logs.update({"T": t, "beta": 1 / (t + 10**-9)}) 

        #     check_stuck_steps = 2**16 // logs["size"] + 1
        #     if (self.step < self.n_wait_steps and self.step % check_stuck_steps == 0 and
        #         logs["kappa_loss"] < -2.53):
        #         print("\nModel seems stuck, reinitializing...")
        #         for layer in self.model.layers: 
        #             for k, initializer in layer.__dict__.items():
        #                 if "initializer" not in k:
        #                     continue
        #                 var = getattr(layer, k.replace("_initializer", ""))
        #                 var.assign(initializer(var.shape, var.dtype))

    def on_epoch_end(self, epoch, logs={}):
        t = float(K.get_value(self.T))
        if logs is not None:
            logs.update({"T": t, "beta": 1 / (t + 10**-9)})
        else:
            logs = {"T": t, "beta": 1 / (t + 10**-9)}

    def get_config(self):
        return {"name": self.name,
                "T": float(K.get_value(self.T))}


class AutomaticTemperatureSchedule(ConstantTemperatureSchedule):
    """docstring for PiecewiseConstantTemperature"""
    def __init__(self, T_start, T_stop=0.001, decay=0.99, rtol=0.05,
        min_lr=0.001, n_checkpoints=10, n_average=100, out_dir="", **kwargs):

        self.T_start = float(K.get_value(T_start))

        assert self.T_start > T_stop

        self.T_stop = T_stop
        self.decay = decay
        self.rtol = rtol
        self.min_lr = min_lr
        self.n_checkpoints = n_checkpoints
        self.n_average = n_average
        self.out_dir = out_dir

        self.T_save = np.geomspace(self.T_start, T_stop, n_checkpoints)

        self.prev_metric = {}

        super(AutomaticTemperatureSchedule, self).__init__(T_start, **kwargs)


    def on_train_batch_end(self, batch, logs={}):

        current = {}
        for metric, value in logs.items():
            if metric not in ["batch", "size"]:
                if batch == 0:
                    current[metric + "_current"] = value
                else:
                    current[metric + "_current"] = (batch + 1) * (
                        value - batch / (batch + 1) * self.prev_metric[metric]
                    )

                self.prev_metric[metric] = value

        logs.update(current)

        t = self.schedule(batch, logs)
        K.set_value(self.T, t)

        logs.update({"T": t, "beta": 1 / (t + 10**-9)}) 


    def schedule(self, step, logs={}):

        Told = np.round(float(K.get_value(self.T)), 6)

        if np.isclose(
            logs["kappa_mean_current"], - logs["fvm_mean_neg_dot_prod_current"] / Told,
            rtol=self.rtol):

            Tnew = self.decay * Told

            if any((Told >= self.T_save) & (Tnew <= self.T_save)):
                self._save_model(Told)

            if Tnew <= self.T_stop:
                self.model.stop_training = True
                return Tnew

            #self._set_lr(Tnew, logs)

            return Tnew
        else:
            return Told


    def _save_model(self, T):
        model_path = "model_T={:5.4f}.h5".format(T)
        model_path = os.path.join(self.out_dir, model_path)
        self.model.save(model_path)


    def _set_lr(self, T, logs):
        lr = float(K.get_value(self.model.optimizer.lr))
        lr = max(lr * (T / self.T_start), self.min_lr)
        K.set_value(self.model.optimizer.lr, lr)
        if logs is not None:
            logs.update({"lr": lr})
        else:
            logs = {"lr": lr}
      

    def get_config(self):
        config = super(AutomaticTemperatureSchedule, self).get_config()
        config.update({
            "T_start": self.T_start,
            "T_stop": self.T_stop,
            "decay": self.decay,
            "rtol": self.rtol,
            "min_lr": self.min_lr,
            "n_checkpoints": self.n_checkpoints,
            "out_dir": self.out_dir,
            "name": self.name
            })
        return config