import os
import argparse
import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

import shutil
import yaml

from hashlib import md5
from tensorflow.keras.layers import (Input, Reshape, Dropout, BatchNormalization, Lambda, Dense)
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import backend as K
from multiprocessing import cpu_count
from GPUtil import getFirstAvailable


class EntrackConditional(object):
    """docstring for EntrackConditional"""
    model_name="EntrackConditional"

    def __init__(self, input_shape):

        inputs = Input(shape=input_shape, name="inputs")

        self.keras = tf.keras.Model(
            inputs,
            self.model_fn(inputs),
            name=self.model_name
        )

    @staticmethod
    def model_fn(inputs):
        """MLP with two output heads for mu and kappa"""
        x = Dense(2048, activation="relu")(inputs)
        x = Dense(2048, activation="relu")(x)
        x = Dense(2048, activation="relu")(x)

        mu = Dense(1024, activation="relu")(x)
        mu = Dense(3, activation="linear")(mu)
        mu = Lambda(lambda t: K.l2_normalize(t, axis=-1), name="mu")(mu)

        kappa = Dense(1024, activation="relu")(x)
        kappa = Dense(1, activation="relu")(kappa)
        kappa = Lambda(lambda t: K.squeeze(t, 1), name="kappa")(kappa)

        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda params: tfd.VonMisesFisher(
                mean_direction=params[0], concentration=params[1]),
            convert_to_tensor_fn=tfd.Distribution.mean
        )([mu, kappa])

    @staticmethod
    def loss(observed_y, predicted_distribution):
        """Negative log-likelihood"""
        return -K.mean(predicted_distribution.log_prob(observed_y))

    def compile(self, optimizer):
        self.keras.compile(optimizer=optimizer, loss=self.loss)


MODELS = {"EntrackConditional": EntrackConditional}


class ConditionalSamples(tf.keras.utils.Sequence):
    def __init__(self,
                 sample_path,
                 batch_size=256,
                 istraining=True,
                 max_n_samples=np.inf):
        """"""
        self.batch_size = batch_size
        self.istraining = istraining

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


def train(model_name,
          train_path,
          eval_path,
          max_n_samples,
          batch_size,
          epochs,
          learning_rate,
          optimizer,
          out_dir):

    input_shape = tuple(np.load(train_path)["input_shape"])
    model = MODELS[model_name](input_shape)
    model.keras.summary()

    # Run Training

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    out_dir = os.path.join(out_dir, model_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    train_seq = ConditionalSamples(
        train_path,
        batch_size,
        max_n_samples=max_n_samples
    )
    callbacks = [
        TensorBoard(log_dir=out_dir,
                    write_graph=False,
                    update_freq=5 * batch_size,
                    profile_batch=0)
    ]

    if eval_path is not None:
        eval_seq = ConditionalSamples(eval_path, istraining=False)
        callbacks.append(
            ModelCheckpoint(os.path.join(out_dir,
                "weights.{epoch:02d}-{val_loss:.2f}.h5"), save_best_only=True)
        )
    else:
        eval_seq = None

    try:
        no_exception = True

        optimizer=getattr(tf.keras.optimizers, optimizer)(learning_rate)
        model.compile(optimizer)

        train_history = model.keras.fit_generator(
            train_seq,
            epochs=epochs,
            validation_data=eval_seq,
            callbacks=callbacks,
            max_queue_size=2 * batch_size,
            use_multiprocessing=True,
            workers=cpu_count()
        )
    except KeyboardInterrupt:
        os.rename(out_dir, out_dir + "_stopped")
        out_dir = out_dir + "_stopped"
    except Exception as e:
        shutil.rmtree(out_dir)
        no_exception = False
        raise e
    finally:
        if no_exception:
            config_path = os.path.join(out_dir, "config" + ".yml")
            print("Saving {}".format(config_path))
            with open(config_path, "w") as file:
                yaml.dump(config, file, default_flow_style=False)

            if eval_path is None:
                model_path = os.path.join(out_dir, "model.h5")
                print("Saving {}".format(model_path))
                model.keras.save(model_path)

    return model


if __name__ == '__main__':

    os.environ['PYTHONHASHSEED'] = '0'
    tf.compat.v1.set_random_seed(3)
    np.random.seed(3)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(getFirstAvailable(
            order="load", maxLoad=10 ** -6, maxMemory=10 ** -1)[0])
    except Exception as e:
        print(str(e))

    parser = argparse.ArgumentParser(description="Train a fiber tracking model")

    parser.add_argument("model_name", type=str, choices=list(MODELS.keys()),
        help="Name of model to be trained.")

    parser.add_argument("train_path", type=str,
        help="Path to training samples file generated by `generate_conditional_samples.py` are saved")

    parser.add_argument("--eval", type=str, default=None, dest="eval_path",
        help="Path to evaluation samples file generated by `generate_conditional_samples.py` are saved")

    parser.add_argument("--max_n_samples", type=int, default=np.inf,
        help="Maximum number of samples to be used for both training and evaluation")

    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size")

    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")

    parser.add_argument("--lr", type=float, default=0.001, dest="learning_rate",
                        help="Learning rate.")

    parser.add_argument("--opt", type=str, default="Adam", dest="optimizer",
                        help="Optimizer name.")

    parser.add_argument("--out", type=str, default='models', dest="out_dir",
        help="Directory to save the training results")

    args = parser.parse_args()

    train(args.model_name,
          args.train_path,
          args.eval_path,
          args.max_n_samples,
          args.batch_size,
          args.epochs,
          args.learning_rate,
          args.optimizer,
          args.out_dir)