import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import Model
from tensorflow.python.keras.utils import Sequence


class Summaries(TensorBoard):

    def __init__(self, *args, _for=None, eval_seq=None, output_freq=0, **kwargs):
        super(Summaries, self).__init__(*args, **kwargs)
        
        if (_for is not None) and (eval_seq is None):
            raise ValueError("Please provide eval_seq")
        if (eval_seq is not None) and (_for is None):
            raise ValueError("Please provide _for")

        self._for = _for
        self.eval_seq = eval_seq
        self.output_freq = output_freq

    def set_model(self, model):

        super(Summaries, self).set_model(model)

        outputs = [layer.output for layer in model.layers
            if layer.name in self._for]

        self.output_model = Model(model.input, outputs)

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        super(Summaries, self).on_epoch_end(epoch, logs)

        step = epoch if self.update_freq == 'epoch' else self._samples_seen

        if self.output_freq and epoch % self.output_freq == 0:
            self._log_activations(epoch)

    def _log_activations(self, epoch):
        """Logs the outputs of the Model to TensorBoard."""
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(), writer.as_default(), \
            summary_ops_v2.always_record_summaries():
            
            outputs = self.output_model.predict_generator(self.eval_seq)

            if not isinstance(outputs, list):
                outputs = [outputs]

            for i, output in enumerate(outputs):
                output_name = self._for[i]
                summary_ops_v2.histogram(output_name, output, step=epoch)
                summary_ops_v2.scalar(output_name+"_mean",
                    np.mean(output), step=epoch)

            writer.flush()


class ConditionalSamples(Sequence):
    def __init__(self,
                 sample_path,
                 batch_size=256,
                 istraining=True,
                 max_n_samples=np.inf):
        """"""
        self.batch_size = batch_size
        self.istraining = istraining

        print("Loading {} samples...".format("train" if istraining else "eval"))
        samples = np.load(sample_path)

        self.inputs = samples["inputs"]
        self.outputs = samples["outputs"]

        assert len(self.inputs) == len(self.outputs)

        self.inputs = self.inputs[:min(max_n_samples, len(self.inputs))]
        self.outputs = self.outputs[:min(max_n_samples, len(self.outputs))]

        self.n_samples = len(self.inputs)

        assert self.n_samples > 0

    def __len__(self):
        if self.istraining:
            return self.n_samples // self.batch_size  # drop remainder
        else:
            return np.ceil(self.n_samples / self.batch_size).astype(int)

    def __getitem__(self, idx):
        x_batch = np.vstack(
            self.inputs[idx * self.batch_size:(idx + 1) * self.batch_size])
        y_batch = np.vstack(
            self.outputs[idx * self.batch_size:(idx + 1) * self.batch_size])

        return x_batch, y_batch