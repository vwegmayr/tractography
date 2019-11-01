import os
import argparse
import datetime
import shutil
import yaml
import importlib
import logging

from tensorflow.keras import callbacks
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import (TensorBoard, ModelCheckpoint,
    ReduceLROnPlateau, Callback)
from multiprocessing import cpu_count
from GPUtil import getFirstAvailable

from models import MODELS
from utils import training as T


def train(model_name,
          train_path,
          eval_path,
          max_n_samples,
          batch_size,
          epochs,
          learning_rate,
          optimizer,
          suffix,
          loss_weight,
          temperature,
          out_dir):

    # Load Model ###############################################################

    input_shape = tuple(np.load(train_path, allow_pickle=True)["input_shape"])

    if "Entrack" in model_name:
        temp = T.Temperature(temperature)
        model = MODELS[model_name](input_shape, temp)
    elif "RNN" in model_name:
        model = MODELS[model_name](input_shape, batch_size=batch_size)
    else:
        model = MODELS[model_name](input_shape, loss_weight=loss_weight)

    # Load Sampler #############################################################

    sampler = getattr(T, model.sample_class)

    train_seq = sampler(
        train_path,
        batch_size,
        max_n_samples=max_n_samples
    )

    # Load Callbacks ###########################################################

    if "Entrack" in model_name:
        callbacks = [
            T.ConstantTemperatureSchedule(
            temp
            )
            #T.LinearTemperatureScheduleWithWarmup(
            #T_start=temp,
            #T_warmup=0.01,
            #T_end=0.0001,
            #n_wait_steps=0,
            #n_warmup_steps=0,
            #n_steps=len(train_seq)*epochs
            #)
        ]
    elif "RNN" in model_name:
        callbacks = [T.RNNResetCallBack(train_seq.reset_batches)]
    else:
        callbacks = []

    # Run Training #############################################################
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    out_dir = os.path.join(out_dir, model_name, suffix, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                          factor=0.5,
                          patience=5,
                          min_lr=0.0001)
    )
    if eval_path is not None:
        eval_seq = sampler(
            eval_path,
            max_n_samples=max_n_samples,
            istraining=False)
        callbacks.append(
            ModelCheckpoint(
                os.path.join(out_dir, "model.{epoch:02d}-{val_loss:.2f}.h5"),
                save_best_only=True,
                save_weights_only=False,
                period=5)
        )
    else:
        eval_seq = None

    # Put Tensorboard callback at the end to catch logs from previous callbacks
    if hasattr(model, "summaries"):
        callbacks.append(
            getattr(T, model.summaries)(
            eval_seq=eval_seq,
            activations_freq=len(train_seq)//5, # 5x per epoch
            log_dir=out_dir,
            write_graph=False,
            update_freq=len(train_seq)//10,
            profile_batch=0
            )
        )
    try:
        print("\nStart training, saving to {}\n".format(out_dir))

        no_exception = True

        optimizer=getattr(tf.keras.optimizers, optimizer)(learning_rate, clipnorm=1.0)
        model.compile(optimizer)

        do_shuffle = False if "RNN" in model_name else True
        train_history = model.keras.fit_generator(
            train_seq,
            epochs=epochs,
            validation_data=eval_seq,
            callbacks=callbacks,
            max_queue_size=4 * batch_size,
            use_multiprocessing=True,
            shuffle=do_shuffle,
            workers=cpu_count()
        )
    #except KeyboardInterrupt:
    #    os.rename(out_dir, out_dir + "_stopped")
    #    out_dir = out_dir + "_stopped"
    except Exception as e:
        shutil.rmtree(out_dir)
        no_exception = False
        raise e
    finally:
        if no_exception:
            config=dict(
                model_name=model_name,
                train_path=train_path,
                eval_path=str(eval_path),
                max_n_samples=str(max_n_samples),
                batch_size=str(batch_size),
                epochs=str(epochs),
                learning_rate=str(learning_rate),
                optimizer=optimizer._keras_api_names[0])
            if "Hybrid" in model_name:
                config["loss_weight"] = str(loss_weight)
            if "Entrack" in model_name:
                config["temperature"] = str(temperature)

            config_path = os.path.join(out_dir, "config" + ".yml")
            print("\nSaving {}".format(config_path))
            with open(config_path, "w") as file:
                yaml.dump(config, file, default_flow_style=False)

            if eval_path is None:
                model_path = os.path.join(out_dir, "model.h5")
                print("Saving {}".format(model_path))
                model.keras.save(model_path)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a fiber tracking model")

    parser.add_argument("model_name", type=str, choices=list(MODELS.keys()),
        help="Name of model to be trained.")

    parser.add_argument("train_path", type=str,
        help="Path to training samples file generated by "
        "`generate_conditional_samples.py` are saved")

    parser.add_argument("--eval", type=str, default=None, dest="eval_path",
        help="Path to evaluation samples file generated by "
        "`generate_conditional_samples.py` are saved")

    parser.add_argument("--max_n_samples", type=int, default=np.inf,
        help="Maximum number of samples to be used for both training and "
        "evaluation")

    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size")

    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")

    parser.add_argument("--lr", type=float, default=0.001, dest="learning_rate",
                        help="Learning rate.")

    parser.add_argument("--opt", type=str, default="Adam", dest="optimizer",
                        help="Optimizer name.")

    parser.add_argument("--suffix", type=str, default="",
        help="Model subfolder to distinguish e.g. conditional and prior models.")

    parser.add_argument("--out", type=str, default='models', dest="out_dir",
        help="Directory to save the training results")

    parser.add_argument("--lw", type=float, default=None, dest="loss_weight",
        help="Total weight of terminal loss, must be set for hybrid models.")

    parser.add_argument("--T", type=float, default=None, dest="temperature",
        help="Temperature, must be set for Entrack models.")

    args = parser.parse_args()

    if "Hybrid" in args.model_name:
        if args.loss_weight is None:
            parser.error("Hybrid models require loss_weight (--lw).")

    if "Entrack" in args.model_name:
        if args.temperature is None:
            parser.error("Entrack models require temperature (--T).")

    os.environ['PYTHONHASHSEED'] = '0'
    tf.compat.v1.set_random_seed(3)
    np.random.seed(3)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(getFirstAvailable(order="load",
            maxLoad=10 ** -6, maxMemory=10 ** -1)[0])
    except Exception as e:
        print(str(e))

    train(args.model_name,
          args.train_path,
          args.eval_path,
          args.max_n_samples,
          args.batch_size,
          args.epochs,
          args.learning_rate,
          args.optimizer,
          args.suffix,
          args.loss_weight,
          args.temperature,
          args.out_dir)