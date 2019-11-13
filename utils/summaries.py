import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as colors

from tensorflow.keras import Model
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2

from sklearn.metrics import precision_recall_curve, average_precision_score

class TBSummaries(TensorBoard):

    def __init__(self,
        out_dir=None,
        eval_seq=None,
        update_freq="batch",
        activations_freq=0,
        activations=None,
        scatter=None,
        scatter_freq=0
        ):
        super(TBSummaries, self).__init__(
            log_dir=out_dir,
            update_freq=update_freq,
            write_graph=False,
            profile_batch=0
            )
        if activations_freq and activations is None:
            raise ValueError("activations must be set for activations_freq != 0")

        if scatter_freq and scatter is None:
            raise ValueError("scatter must be set for scatter_freq != 0")

        self.activations_freq = activations_freq
        self.activations = activations

        self.scatter = scatter
        self.scatter_freq = scatter_freq

        self.eval_seq = eval_seq

    def set_model(self, model):
        super(TBSummaries, self).set_model(model)

        if self.activations_freq:
            outputs = [model.get_layer(name).output for name in self.activations]
            self.activation_model = Model(model.input, outputs)

        if self.scatter_freq:
            x = [model.get_layer(name[0]).output for name in self.scatter]
            y = [model.get_layer(name[1]).output for name in self.scatter]
            self.scatter_x_model = Model(model.input, x)
            self.scatter_y_model = Model(model.input, y)

    def on_train_batch_end(self, batch, logs={}):
        super(TBSummaries, self).on_train_batch_end(batch, logs)

        if (self.activations_freq and self.activations_freq != "epoch" and
            self._total_batches_seen % self.activations_freq == 0):
            self._log_activations(self._total_batches_seen)

        if (self.scatter_freq and self.scatter_freq != "epoch" and
            self._total_batches_seen % self.scatter_freq == 0):
            self._scatter(self._total_batches_seen, logs)

    def on_epoch_end(self, epoch, logs={}):
        """Runs metrics and histogram summaries at epoch end."""
        super(TBSummaries, self).on_epoch_end(epoch, logs)

        if self.activations_freq and self.activations_freq == "epoch":
            self._log_activations(epoch)

        if self.scatter_freq and self.scatter_freq == "epoch":
            self._scatter(epoch, logs)

    def _log_activations(self, step):
        """Logs the outputs of the Model to TensorBoard."""
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(), writer.as_default(), \
            summary_ops_v2.always_record_summaries():

            activations = self.activation_model.predict_generator(self.eval_seq)
            if not isinstance(activations, list):
                activations = [activations]

            for i, values in enumerate(activations):
                name = self.activations[i]
                summary_ops_v2.histogram(name, values, step=step)
                summary_ops_v2.scalar(name+"_mean",
                    np.mean(values), step=step)
            writer.flush()

    def _scatter(self, epoch, logs={}):
        x = self.scatter_x_model.predict_generator(self.eval_seq)
        y = self.scatter_y_model.predict_generator(self.eval_seq)
        # =====================================================================
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(), writer.as_default(), \
            summary_ops_v2.always_record_summaries():
            # ------------------------------------------------------------------
            for i, name in self.scatter:
                fig, ax = plt.subplots()
                ax.hist2d(x[i], y[i], bins=50, density=True,
                    norm=colors.SymLogNorm(linthresh=0.01, linscale=2, vmin=-1.0,
                        vmax=2.0))
                ax.set_xlabel(name[0])
                ax.set_ylabel(name[1])
                fig.canvas.draw()
                # ------------------------------------------------------------------
                plot = np.array(fig.canvas.renderer.buffer_rgba())
                plot = np.expand_dims(plot, 0)
                summary_ops_v2.image("2DHistogram", plot, step=step)
                # ------------------------------------------------------------------
                plt.close()
                writer.flush()


class FvMSummaries(TBSummaries):

    def __init__(self, *args,
        eval_seq=None,
        activations_freq="epoch",
        activations=["kappa"],
        **kwargs):
        super(FvMSummaries, self).__init__(*args,
            eval_seq=eval_seq,
            activations=activations,
            activations_freq=activations_freq,
            **kwargs)


class FvMHybridSummaries(TBSummaries):

    def __init__(self, *args,
        eval_seq=None,
        activations_freq="epoch",
        activations=["kappa", "mu", "isterminal"],
        **kwargs):
        super(FvMHybridSummaries, self).__init__(*args,
            eval_seq=eval_seq,
            activations_freq=activations_freq,
            activations=activations,
            **kwargs)

    def _log_activations(self, step):
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
                step=step)
            summary_ops_v2.scalar("kappa_midway_mean",
                np.mean(kappa[midway]), step=step)
            # ------------------------------------------------------------------
            summary_ops_v2.histogram("kappa_terminal", kappa[terminal],
                step=step)
            summary_ops_v2.scalar("kappa_terminal_mean",
                np.mean(kappa[terminal]), step=step)
            # ==================================================================
            mu = activation_values[1]
            neg_dot_prod_midway = -np.sum(
                mu[midway] * self.eval_seq.outgoing[midway], axis=1)
            summary_ops_v2.histogram("neg_dot_prod_midway", neg_dot_prod_midway,
                step=step)
            summary_ops_v2.scalar("neg_dot_prod_midway_mean",
                np.mean(neg_dot_prod_midway), step=step)
            # ------------------------------------------------------------------
            neg_dot_prod_terminal = -np.sum(
                mu[terminal] * self.eval_seq.outgoing[terminal], axis=1)
            summary_ops_v2.histogram("neg_dot_prod_terminal",
                neg_dot_prod_terminal, step=step)
            summary_ops_v2.scalar("neg_dot_prod_terminal_mean",
                np.mean(neg_dot_prod_terminal), step=step)
            # ==================================================================
            isterminal = activation_values[2]
            ave_prec = average_precision_score(self.eval_seq.isterminal, isterminal)
            summary_ops_v2.scalar("average_precision", ave_prec, step=step)
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
            summary_ops_v2.image("Precision-Recall", plot, step=step)
            # ==================================================================
            plt.close()
            writer.flush()


class EntrackSummaries(TBSummaries):

    def __init__(self,
                 out_dir=None,
                 eval_seq=None,
                 update_freq="batch",
                 activations=["kappa"],
                 activations_freq="epoch",
                 scatter=[("kappa", "mu")],
                 scatter_freq="epoch"
        ):
        super(EntrackSummaries, self).__init__(
            out_dir=out_dir,
            eval_seq=eval_seq,
            update_freq=update_freq,
            activations_freq=activations_freq,
            activations=activations,
            scatter=scatter,
            scatter_freq=scatter_freq,
        )

    def _scatter(self, step, logs):
        kappa_pred = self.scatter_x_model.predict_generator(self.eval_seq)
        mu_pred = self.scatter_y_model.predict_generator(self.eval_seq)

        mu_true = self.eval_seq.outgoing

        agreement = np.sum(np.squeeze(mu_true) * mu_pred, axis=1)
        kappa_mean = kappa_pred.mean() + 10**-9
        kappa_pred /= kappa_mean

        # =====================================================================
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(), writer.as_default(), \
            summary_ops_v2.always_record_summaries():
            # ------------------------------------------------------------------
            fig, ax = plt.subplots()
            ax.hist2d(kappa_pred, agreement, bins=50, density=True,
                norm=colors.SymLogNorm(linthresh=0.01, linscale=2, vmin=-1.0,
                    vmax=2.0))
            ax.plot([0, 1/(logs["T"]+10**-9)/kappa_mean], [0, 1], color="red")
            ax.set_xlabel("Certainty k")
            ax.set_ylabel("Agreement <m,v>")
            fig.canvas.draw()
            # ------------------------------------------------------------------
            plot = np.array(fig.canvas.renderer.buffer_rgba())
            plot = np.expand_dims(plot, 0)
            summary_ops_v2.image("2DHistogram", plot, step=step)
            # ------------------------------------------------------------------
            plt.close()
            writer.flush()


class RNNSummaries(TensorBoard):

    def __init__(self, *args,
        out_dir=None,
        eval_seq=None,
        activations=None,
        activations_freq=0,
        **kwargs):
        super(RNNSummaries, self).__init__(*args, log_dir=out_dir, **kwargs)

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