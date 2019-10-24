import os
import argparse

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

os.environ['PYTHONHASHSEED'] = '0'
tf.compat.v1.set_random_seed(42)
np.random.seed(42)

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getFirstAvailable(
        order="load", maxLoad=10 ** -6, maxMemory=10 ** -1)[0])
except Exception as e:
    print(str(e))


class ConditionalSamples(tf.keras.utils.Sequence):
    def __init__(self, sample_dir, batch_size=256, istraining=True, max_n_samples=np.inf):
        """"""
        self.batch_size = batch_size
        self.istraining = istraining

        samples = np.load(os.path.join(sample_dir, "samples.npz"))

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
        x_batch = np.vstack(self.inputs[idx * self.batch_size:(idx + 1) * self.batch_size])
        y_batch = np.vstack(self.outputs[idx * self.batch_size:(idx + 1) * self.batch_size])

        return x_batch, y_batch


def train_model(config):

    hasher = md5()
    for v in config.values():
        hasher.update(str(v).encode())

    save_dir = os.path.join(config['out_dir'], config["model_name"], hasher.hexdigest())

    if os.path.exists(save_dir):
        print("This model config has been trained already:\n{}".format(save_dir))
        return

    # Define Model Function and Loss

    train_path = os.path.join(config["train_dir"], "samples.npz")

    input_shape = np.load(train_path)["inputs"].shape[1:]
    inputs = Input(shape=input_shape, name="inputs")

    def model_fn(inputs):

        x = Dense(1024, activation="relu")(inputs)

        x = Dense(1024, activation="relu")(x)

        mu = Dense(512, activation="relu")(x)
        mu = Dense(3, activation="linear")(mu)
        mu = Lambda(lambda t: K.l2_normalize(t, axis=-1), name="mu")(mu)

        kappa = Dense(512, activation="relu")(x)
        kappa = Dense(1, activation="relu")(kappa)
        kappa = Lambda(lambda t: K.squeeze(t, 1), name="kappa")(kappa)

        return tfp.layers.DistributionLambda(
            make_distribution_fn=lambda params: tfd.VonMisesFisher(mean_direction=params[0],
                                                                   concentration=params[1]),
            convert_to_tensor_fn=tfd.Distribution.mean
        )([mu, kappa])

    model = tf.keras.Model(inputs, model_fn(inputs), name=config["model_name"])
    model.summary()

    def negative_log_likelihood(observed_y, predicted_distribution):
        return -K.mean(predicted_distribution.log_prob(observed_y))

    # Run Training

    train_seq = ConditionalSamples(config["train_dir"], config["batch_size"],
                                    max_n_samples=config["max_n_samples"])
    eval_seq = None
    if config['eval_dir'] is not None:
        eval_seq = ConditionalSamples(config["eval_dir"], istraining=False)
    try:
        no_exception = True

        os.makedirs(save_dir, exist_ok=True)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=negative_log_likelihood
        )
        train_history = model.fit_generator(
            train_seq,
            epochs=config["epochs"],
            validation_data=eval_seq,
            callbacks=[
                TensorBoard(log_dir=save_dir,
                            write_graph=False,
                            update_freq=5 * config["batch_size"],
                            profile_batch=0),
            ],
            max_queue_size=2 * config["batch_size"],
            use_multiprocessing=True,
            workers=cpu_count()
        )
    except KeyboardInterrupt:
        os.rename(save_dir, save_dir + "_stopped")
        save_dir = save_dir + "_stopped"
    except Exception as e:
        shutil.rmtree(save_dir)
        no_exception = False
        raise e
    finally:
        if no_exception:
            config_path = os.path.join(save_dir, "config" + ".yml")
            print("Saving {}".format(config_path))
            with open(config_path, "w") as file:
                yaml.dump(config, file, default_flow_style=False)

            model_path = os.path.join(save_dir, "model.h5")
            print("Saving {}".format(model_path))
            model.save(model_path)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train the entrack model")

    parser.add_argument("train_dir", type=str,
        help="Path to the dir where training samples generated by `generate_conditional_samples.py` are saved")

    parser.add_argument("--eval_dir", type=str, default=None,
        help="Path to the dir where evaluation samples generated by `generate_conditional_samples.py` are saved")

    parser.add_argument("--max_n_samples", type=int, default=np.inf,
        help="Maximum number of samples to be used for both training and evaluation")

    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size")

    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")

    parser.add_argument("--out_dir", type=str, default='../models/',
        help="Directory to save the training results")

    args = parser.parse_args()

    config = dict(
        model_name="entrack_conditional",
        train_dir=args.train_dir,
        eval_dir=args.eval_dir,
        max_n_samples=args.max_n_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        out_dir=args.out_dir
    )

    train_model(config)