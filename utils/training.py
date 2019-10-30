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
        self.samples = np.load(sample_path, allow_pickle=True)  # lazy loading
        self.n_samples = min(max_n_samples, self.samples["n_samples"])

    def __len__(self):
        if self.istraining:
            return self.n_samples // self.batch_size  # drop remainder
        else:
            return np.ceil(self.n_samples / self.batch_size).astype(int)


class FvMSamples(Samples):

    def __init__(self, *args, **kwargs):
        super(FvMSamples, self).__init__(*args, **kwargs)
        print("Loading {} samples...".format("train" if self.istraining else "eval"))
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


class RNNSamples(Samples):

    def __init__(self, *args, **kwargs):
        super(RNNSamples, self).__init__(*args, **kwargs)
        print("RNNSamples: Loading {} samples...".format("train" if self.istraining else "eval"))

        # Cut whatever doesn't fit in a batch
        if self.batch_size > 1:
            self.inputs = np.array([batch_input[:-(batch_input.shape[0] % self.batch_size),...]
                                    if batch_input.shape[0] > self.batch_size else batch_input
                                    for batch_input in self.samples["inputs"] ])
            self.outgoing = np.array([batch_input[:-(batch_input.shape[0] % self.batch_size), ...]
                                      if batch_input.shape[0] > self.batch_size else batch_input
                                      for batch_input in self.samples["outgoing"]])
        self.inputs = np.array([batch_input for batch_input in self.inputs if batch_input.shape[0] >= self.batch_size])
        self.outgoing = np.array([batch_input for batch_input in self.outgoing if batch_input.shape[0] >= self.batch_size])

        print("Reduced length of input from {0} to {1} to fit the batch size.".
              format(len(self.samples["inputs"]), len(self.inputs)))

        # To help find the right fiber for the right batch index
        self.batch_indices = np.cumsum([(fiber.shape[0] // self.batch_size) * fiber.shape[1] for fiber in self.inputs])
        self.reset_batches = self._get_reset_batches()

    def __len__(self):
        return self.batch_indices[-1]

    def __getitem__(self, idx):
        first_possible_input = self.inputs[(idx < self.batch_indices)][0]
        first_possible_output = self.outgoing[(idx < self.batch_indices)][0]
        previous_index = np.where((idx < self.batch_indices))[0][0] - 1
        if previous_index < 0:
            previous_index = 0
        else:
            previous_index = self.batch_indices[previous_index]
        current_batch_idx = idx - previous_index
        row_idx = current_batch_idx // first_possible_input.shape[1]
        col_idx = current_batch_idx % first_possible_input.shape[1]

        x_batch = first_possible_input[row_idx * self.batch_size:(row_idx + 1) * self.batch_size, col_idx, np.newaxis, ...]
        y_batch = first_possible_output[row_idx * self.batch_size:(row_idx + 1) * self.batch_size, col_idx, np.newaxis, ...]

        reset_state = False
        if col_idx == 0:
            reset_state = True

        return (x_batch, y_batch)

    def _get_reset_batches(self):
        """For sure there is a smarter way for this ... :)"""
        reset_batches = []
        for idx in range(self.__len__()):
            first_possible_input = self.inputs[(idx < self.batch_indices)][0]
            previous_index = np.where((idx < self.batch_indices))[0][0] - 1
            if previous_index < 0:
                previous_index = 0
            else:
                previous_index = self.batch_indices[previous_index]
            current_batch_idx = idx - previous_index
            col_idx = current_batch_idx % first_possible_input.shape[1]
            if col_idx == 0:
                reset_batches.append(idx)
        return np.array(reset_batches)


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

    def __init__(self, *args, activations=["kappa"], activations_freq=1, **kwargs):
        super(FvMSummaries, self).__init__(*args, activations=activations,
            activations_freq=activations_freq, **kwargs)


class FvMHybridSummaries(TBSummaries):

    def __init__(self, *args, activations=["kappa", "mu", "isterminal"], activations_freq=1, **kwargs):
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


class RNNSummaries(TensorBoard):

    def __init__(self, *args, eval_seq=None, activations=None, activations_freq=0, **kwargs):
        super(RNNSummaries, self).__init__(*args, **kwargs)

        if activations_freq > 0 and activations is None:
            raise ValueError("activations must be set for activations_freq > 0")

        self.activations_freq = activations_freq
        self.activations = activations
        self.eval_seq = eval_seq

    def set_model(self, model):
        super(RNNSummaries, self).set_model(model)
        if self.activations_freq > 0:
            outputs = [model.get_layer(name).output for name in self.activations]
            self.activation_model = Model(model.input, outputs)

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        super(RNNSummaries, self).on_epoch_end(epoch, logs)
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


class RNNResetCallBack(Callback):
    def __init__(self, reset_batches):
        super(RNNResetCallBack, self).__init__()
        self.reset_batches = reset_batches

    def on_batch_end(self, batch, logs={}):
        if batch in self.reset_batches:
            self.model.reset_states()
        return


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
    def __init__(self, T_start, T_end, n_steps, **kwargs):
        self.T_start = float(K.get_value(T_start))
        self.decay_rate = (self.T_start - T_end) / (n_steps * T_end)
        super(HarmonicTemperatureSchedule, self).__init__(T_start, **kwargs)

    def schedule(self, step):
        return self.T_start / (1 + self.decay_rate * step)

    def get_config(self):
        config = super(HarmonicTemperatureSchedule, self).get_config()
        config["decay_rate"] = self.decay_rate
        config["T_start"] = self.T_start
        config["name"] = self.name
        return config