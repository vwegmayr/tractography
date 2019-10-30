import numpy as np

from matplotlib import pyplot as plt

from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras.callbacks import TensorBoard, Callback
from tensorflow.keras import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from sklearn.metrics import precision_recall_curve, average_precision_score


class Samples(Sequence):
    def __init__(self,
                 sample_path,
                 batch_size=256,
                 istraining=True,
                 max_n_samples=np.inf):
        """"""
        self.batch_size = batch_size
        self.istraining = istraining
        self.samples = np.load(sample_path) # lazy loading
        self.n_samples = min(max_n_samples, self.samples["n_samples"])

    def __len__(self):
        if self.istraining:
            return self.n_samples // self.batch_size  # drop remainder
        else:
            return np.ceil(self.n_samples / self.batch_size).astype(int)


class FvMSamples(Samples):

    def __init__(self, *args, **kwargs):
        super(FvMSamples, self).__init__(*args, **kwargs)
        print("Loading {} samples...".format("train" if self.istraining
            else "eval"))
        self.inputs = self.samples["inputs"]
        self.outgoing = self.samples["outgoing"]

    def __getitem__(self, idx):
        x_batch = self.inputs[idx*self.batch_size:(idx + 1)*self.batch_size]
        y_batch = self.outgoing[idx*self.batch_size:(idx + 1)*self.batch_size]
        return x_batch, y_batch


class EntrackSamples(FvMSamples):
    """docstring for EntrackSamples"""
    
    def __getitem__(self, idx):
        inputs, outgoing = super(EntrackSamples, self).__getitem__(idx)
        
        return inputs, {"fvm": outgoing, "kappa": np.zeros(len(outgoing))}


class FvMHybridSamples(FvMSamples):

    def __init__(self, *args, **kwargs):
        super(FvMHybridSamples, self).__init__(*args, **kwargs)
        self.isterminal = self.samples["isterminal"]

    def __getitem__(self, idx):
        inputs = self.inputs[idx*self.batch_size:(idx + 1)*self.batch_size]
        outgoing = self.outgoing[idx*self.batch_size:(idx + 1)*self.batch_size]
        isterminal = self.isterminal[idx*self.batch_size:(idx + 1)*self.batch_size]

        fvm_sample_weights = 1.0 - isterminal
        fvm_sample_weights /= np.sum(fvm_sample_weights)

        n_terminal = np.sum(isterminal)

        if n_terminal > 0:
            isterminal_sample_weights = (
                isterminal*np.sum(1-isterminal) + (1-isterminal)*np.sum(isterminal)
            )
            isterminal_sample_weights /= np.sum(isterminal_sample_weights)
        else:
            isterminal_sample_weights = (
                (1 - isterminal) / (2 * self.batch_size * 0.975)
            )
        return (
            inputs,
            {"fvm": outgoing, "isterminal": isterminal},
            {"fvm": fvm_sample_weights, "isterminal": isterminal_sample_weights}
        )


class TBSummaries(TensorBoard):

    def __init__(self, *args, eval_seq=None, activations=None,
        activations_freq=0, **kwargs):
        super(TBSummaries, self).__init__(*args, **kwargs)

        if activations_freq > 0 and activations is None:
            raise ValueError("activations must be set for activations_freq > 0")

        self.activations_freq = activations_freq
        self.activations = activations
        self.eval_seq = eval_seq

    def set_model(self, model):
        super(TBSummaries, self).set_model(model)
        if self.activations_freq > 0:
            outputs = [model.get_layer(name).output for name in self.activations]
            self.activation_model = Model(model.input, outputs)

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        super(TBSummaries, self).on_epoch_end(epoch, logs)
        if self.activations_freq and epoch % self.activations_freq == 0:
            self._log_activations(epoch)

    def _log_activations(self, epoch):
        """Logs the outputs of the Model to TensorBoard."""
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(), writer.as_default(), \
            summary_ops_v2.always_record_summaries():

            activations = self.activation_model.predict_generator(self.eval_seq)
            if not isinstance(activations, list):
                activations = [activations]
            for i, values in enumerate(activations):
                name = self.activations[i]
                summary_ops_v2.histogram(name, values, step=epoch)
                summary_ops_v2.scalar(name+"_mean",
                    np.mean(values), step=epoch)
            writer.flush()


class FvMSummaries(TBSummaries):

    def __init__(self, *args, activations=["kappa"], activations_freq=1,
        **kwargs):
        super(FvMSummaries, self).__init__(*args, activations=activations,
            activations_freq=activations_freq, **kwargs)


class FvMHybridSummaries(TBSummaries):

    def __init__(self, *args, activations=["kappa", "mu", "isterminal"],
        activations_freq=1, **kwargs):
        super(FvMHybridSummaries, self).__init__(*args, activations=activations,
            activations_freq=activations_freq, **kwargs)

    def _log_activations(self, epoch):
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(), writer.as_default(), \
            summary_ops_v2.always_record_summaries():
            # ==================================================================
            activation_values = self.activation_model.predict_generator(
                self.eval_seq)
            terminal = (self.eval_seq.isterminal == 1)
            midway = np.logical_not(terminal)
            # ==================================================================
            kappa = activation_values[0]
            summary_ops_v2.histogram("kappa_midway", kappa[midway],
                step=epoch)
            summary_ops_v2.scalar("kappa_midway_mean",
                np.mean(kappa[midway]), step=epoch)
            # ------------------------------------------------------------------
            summary_ops_v2.histogram("kappa_terminal", kappa[terminal],
                step=epoch)
            summary_ops_v2.scalar("kappa_terminal_mean",
                np.mean(kappa[terminal]), step=epoch)
            # ==================================================================
            mu = activation_values[1]
            neg_dot_prod_midway = -np.sum(
                mu[midway] * self.eval_seq.outgoing[midway], axis=1)
            summary_ops_v2.histogram("neg_dot_prod_midway", neg_dot_prod_midway,
                step=epoch)
            summary_ops_v2.scalar("neg_dot_prod_midway_mean",
                np.mean(neg_dot_prod_midway), step=epoch)
            # ------------------------------------------------------------------
            neg_dot_prod_terminal = -np.sum(
                mu[terminal] * self.eval_seq.outgoing[terminal], axis=1)
            summary_ops_v2.histogram("neg_dot_prod_terminal",
                neg_dot_prod_terminal, step=epoch)
            summary_ops_v2.scalar("neg_dot_prod_terminal_mean",
                np.mean(neg_dot_prod_terminal), step=epoch)
            # ==================================================================
            isterminal = activation_values[2]
            ave_prec = average_precision_score(self.eval_seq.isterminal, isterminal)
            summary_ops_v2.scalar("average_precision", ave_prec, step=epoch)
            # ------------------------------------------------------------------
            precision, recall, thresh = precision_recall_curve(
                y_true=self.eval_seq.isterminal,
                probas_pred=np.round(isterminal / 0.05) * 0.05
            )
            # ------------------------------------------------------------------
            fig, ax = plt.subplots()
            ax.plot(recall, precision, "-o")
            frac = np.mean(self.eval_seq.isterminal)
            ax.plot([0,1],[frac, frac])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            fig.canvas.draw()
            # ------------------------------------------------------------------
            plot = np.array(fig.canvas.renderer.buffer_rgba())
            plot = np.expand_dims(plot, 0)
            summary_ops_v2.image("Precision-Recall", plot, step=epoch)
            # ==================================================================
            plt.close()
            writer.flush()


class Temperature(ResourceVariable):
    """docstring for Temperature"""
    def __init__(self, value, name="temperature"):
        super(Temperature, self).__init__(value, name=name)

    def get_config(self):
        return {"T": float(K.get_value(self))}


class ConstantTemperatureSchedule(Callback):
    """docstring for ConstantTemperatureSchedule"""
    def __init__(self, temperature, *args, **kwargs):
        super(ConstantTemperatureSchedule, self).__init__(*args, **kwargs)
        self.temperature = temperature
        self.step = 0

    def schedule(self, step):
        return float(K.get_value(self.temperature))

    def on_train_batch_begin(self, batch, logs={}):
        T = self.schedule(self.step)
        K.set_value(self.temperature, T)
        if logs is not None:
            logs.update({"T": T}) 
        else:
            logs = {"T": T}

    def on_train_batch_end(self, batch, logs={}):
        self.step += 1

    def get_config(self):
        return {"name": self.name,
                "temperature": float(K.get_value(self.temperature))}


class PiecewiseConstantTemperatureSchedule(ConstantTemperatureSchedule):
    """docstring for PiecewiseConstantTemperature"""
    def __init__(self, temperature, boundaries, values, **kwargs):

        if values[0] != float(K.get_value(temperature)):
            print("\nWARNING: Provided temperature is not the same as the "
                "first value of PiecewiseConstantTemperatureSchedule.\nUsing "
                "value from PiecewiseConstantTemperatureSchedule.\n")

        super(PiecewiseConstantTemperatureSchedule, self).__init__(temperature,
            **kwargs)
        self.boundaries = boundaries
        self.values = values

    def schedule(self, step):
        for i, b in enumerate(self.boundaries):
            if step <= b:
                return self.values[i]
        if step > self.boundaries[-1]:
            return self.values[-1]

    def get_config(self):
        config = super(PiecewiseConstantTemperatureSchedule, self).get_config()
        config["boundaries"] = self.boundaries
        config["values"] = self.values
        config["name"] = self.name
        return config


class HarmonicTemperatureSchedule(ConstantTemperatureSchedule):
    """docstring for PiecewiseConstantTemperature"""
    def __init__(self, temperature, decay_rate=1.0, **kwargs):
        self.T0 = float(K.get_value(temperature))
        self.decay_rate = decay_rate
        super(HarmonicTemperatureSchedule, self).__init__(temperature, **kwargs)

    def schedule(self, step):
        return self.T0 / (1 + self.decay_rate * step)

    def get_config(self):
        config = super(HarmonicTemperatureSchedule, self).get_config()
        config["decay_rate"] = self.decay_rate
        config["T0"] = self.T0
        config["name"] = self.name
        return config