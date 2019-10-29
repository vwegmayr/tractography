import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import Model
from tensorflow.python.keras.utils import Sequence

from sklearn.metrics import recall_score, precision_score, roc_auc_score


class TBSummaries(TensorBoard):

    def __init__(self, *args, eval_seq=None, activations=None, activations_freq=0, **kwargs):
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


class Samples(Sequence):
    def __init__(self,
                 sample_path,
                 batch_size=256,
                 istraining=True,
                 max_n_samples=np.inf):
        """"""
        self.batch_size = batch_size
        self.istraining = istraining
        self.samples = np.load(sample_path, allow_pickle=True) # lazy loading
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


class RNNSamples(Samples):

    def __init__(self, *args, **kwargs):
        super(RNNSamples, self).__init__(*args, **kwargs)
        print("RNNSamples: Loading {} samples...".format("train" if self.istraining else "eval"))
        self.inputs = self.samples["inputs"]
        self.outgoing = self.samples["outgoing"]

        # Cut whatever doesn't fit in a batch
        self.inputs = np.array([batch_input[:-(batch_input.shape[0] % self.batch_size),...] for batch_input in self.inputs])
        self.outgoing = np.array([batch_input[:-(batch_input.shape[0] % self.batch_size), ...] for batch_input in self.outgoing])

        # To help find the right fiber for the right batch index
        self.batch_indices = np.cumsum([(fiber.shape[0] // self.batch_size) * fiber.shape[1] for fiber in self.inputs])

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

        x_batch = first_possible_input[row_idx * self.batch_size:(row_idx + 1) * self.batch_size, col_idx, ...]
        y_batch = first_possible_output[row_idx * self.batch_size:(row_idx + 1) * self.batch_size, col_idx, ...]

        reset_state = False
        if col_idx == 0:
            reset_state = True

        return x_batch, y_batch, reset_state


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

        isterminal_sample_weights = (1.0 - isterminal) + isterminal * 30.0 
        isterminal_sample_weights /= np.sum(isterminal_sample_weights)

        return (
            inputs,
            {"fvm": outgoing, "isterminal": isterminal},
            {"fvm": fvm_sample_weights, "isterminal": isterminal_sample_weights}
        )


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
            precision = precision_score(self.eval_seq.isterminal, isterminal > 0.5)
            recall = recall_score(self.eval_seq.isterminal, isterminal > 0.5)
            roc_auc = roc_auc_score(self.eval_seq.isterminal, isterminal)
            # ------------------------  ------------------------------------------
            summary_ops_v2.scalar("terminal_precision", precision, step=epoch)
            summary_ops_v2.scalar("terminal_recall", recall, step=epoch)
            summary_ops_v2.scalar("terminal_roc_auc", recall, step=epoch)
            # ==================================================================
            writer.flush()
