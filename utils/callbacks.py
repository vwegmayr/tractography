import os

import numpy as np

from tensorflow.python.keras.callbacks import Callback
from tensorflow.keras import backend as K


class RunningWindowLogger(Callback):
    """docstring for RunningWindowLogger"""
    def __init__(self, metrics, window_size, log_std=False):

        super(RunningWindowLogger, self).__init__()
        self.metrics = metrics
        self.window_size = window_size
        self.log_std = log_std
        self.running_windows = {m: [] for m in metrics}

    def on_batch_end(self, batch, logs={}):
        for k in self.metrics:
            self.running_windows[k].append(float(logs[k]))

            if batch + 1 > self.window_size:
                self.running_windows[k].pop(0)

            if batch + 1 >= self.window_size:
                logs.update({k + "_average": np.mean(self.running_windows[k])})

                if self.log_std:
                    logs.update({k + "_std": np.std(self.running_windows[k])})

        
class RNNResetCallBack(Callback):
    def __init__(self, reset_batches):
        super(RNNResetCallBack, self).__init__()
        self.reset_batches = reset_batches

    def on_batch_end(self, batch, logs={}):
        if batch in self.reset_batches:
            self.model.reset_states()
        return


class AutomaticTemperatureSchedule(Callback):
    """docstring for PiecewiseConstantTemperature"""
    def __init__(self, T_start, T_stop=0.001, decay=0.99, tol=0.05,
        min_lr=None, n_checkpoints=10, reinit_patience=128, out_dir="", **kwargs):

        super(AutomaticTemperatureSchedule, self).__init__()

        self.T = T_start
        self.T_start = float(K.get_value(T_start))
        assert self.T_start > T_stop
        self.T_stop = T_stop
        self.decay = decay
        self.tol = tol
        self.min_lr = min_lr
        self.n_checkpoints = n_checkpoints
        self.out_dir = out_dir
        self.T_save = np.geomspace(self.T_start, T_stop, n_checkpoints)
        self.reinit_patience=reinit_patience
        self._is_stuck_for = 0


    def on_train_batch_end(self, batch, logs={}):

        is_stuck = logs["kappa_mean"] < 0.002
        decreased = False
        Tnow = np.round(float(K.get_value(self.T)), 6)

        self._is_stuck_for += is_stuck

        if self._is_stuck_for > self.reinit_patience:
            self._reinit_model()
            self._is_stuck_for = 0

        if "kappa_mean_average" in logs and not is_stuck:

            if np.isclose(
                logs["kappa_mean_average"],
               -logs["fvm_mean_neg_dot_prod_average"] / Tnow, rtol=self.tol):

                decreased = True

                Tnew = self.decay * Tnow

                K.set_value(self.T, Tnew)

                if any((Tnow >= self.T_save) & (Tnew <= self.T_save)):
                    self._save_model(Tnow)

                if self.min_lr is not None:
                    self._set_lr(Tnew, logs)

                if Tnew <= self.T_stop:
                    self.model.stop_training = True

        t = Tnew if decreased else Tnow
        logs.update({"T": t, "beta": 1 / (t + 10**-9)}) 


    def on_epoch_end(self, epoch, logs={}):
        t = float(K.get_value(self.T))
        logs.update({"T": t, "beta": 1 / (t + 10**-9)})


    def _save_model(self, T):
        model_path = "model_T={:5.4f}.h5".format(T)
        model_path = os.path.join(self.out_dir, model_path)
        self.model.save(model_path)


    def _set_lr(self, T, logs):
        lr = float(K.get_value(self.model.optimizer.lr))
        lr = max(lr * (T / self.T_start), self.min_lr)
        K.set_value(self.model.optimizer.lr, lr)
        logs.update({"lr": lr})


    def _reinit_model(self):
        for layer in self.model.layers: 
            for k, initializer in layer.__dict__.items():
                if "initializer" not in k:
                    continue
                var = getattr(layer, k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))
      

    def get_config(self):
        config = super(AutomaticTemperatureSchedule, self).get_config()
        config.update({
            "T_start": self.T_start,
            "T_stop": self.T_stop,
            "decay": self.decay,
            "tol": self.tol,
            "min_lr": self.min_lr,
            "n_checkpoints": self.n_checkpoints,
            "out_dir": self.out_dir,
            "name": self.name
            })
        return config

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