import os
import argparse
import shutil
import git

from tensorflow.keras import optimizers as keras_optimizers

from models import MODELS
from utils.training import (setup_env, timestamp, parse_callbacks, 
    maybe_get_a_gpu)

import configs

@setup_env
def train(config=None, gpu_queue=None):

    try:
        gpu_idx = maybe_get_a_gpu() if gpu_queue is None else gpu_queue.get()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    except Exception as e:
        print(str(e))

    day, hour = timestamp(separate=True)

    out_dir = os.path.join("models",
                           config["model_name"],
                           config.get("model_type", ""),
                           day,
                           hour)
    
    os.makedirs(out_dir, exist_ok=True)
    configs.deep_update(config, {"out_dir": out_dir})
    
    configs.add(config, to=".running")

    model = MODELS[config["model_name"]](config)

    try:
        train_seq = model.get_sequence(config)
        eval_seq = model.get_sequence(config, istraining=False)
        configs.deep_update(config, {"train_seq": train_seq,
                                     "eval_seq": eval_seq})

        checkpoints = out_dir + '/inter_model_{epoch:02d}-{val_loss:.4f}.h5'
        if "RNN" in config["model_name"]:
            configs.deep_update(config,
                                {"reset_batches": train_seq.reset_batches})
            configs.deep_update(config, {"filepath": checkpoints})
        if 'Trackifier' in config['model_name']:
            configs.deep_update(config, {"filepath": checkpoints})

        callbacks = parse_callbacks(config["callbacks"])
        optimizer=getattr(keras_optimizers, config["optimizer"])(
            **config["opt_params"]
        )

        model.compile(optimizer)

        if isinstance(config['train_path'], list):
            for i, subject in enumerate(config['train_path']):
                samples_config = os.path.join(
                    os.path.dirname(subject), 'config.yml')
                samples_config = configs.load(samples_config)
                config['input_sampels_config_{0}'.format(i)] = samples_config
        else:
            samples_config = os.path.join(
                os.path.dirname(config['train_path']), 'config.yml')
            samples_config = configs.load(samples_config)
            config['input_sampels_config'] = samples_config
        repo = git.Repo(".")
        commit = repo.head.commit
        config['commit'] = str(commit)
        configs.save(config)

        print("\nStart training...")
        no_exception = True
        model.keras.fit_generator(
            train_seq,
            callbacks=callbacks,
            validation_data=eval_seq,
            epochs=config["epochs"],
            shuffle=config["shuffle"],
            max_queue_size=4 * config["batch_size"],
            verbose=1,
            workers=3,
            use_multiprocessing=True,
        )
    except KeyboardInterrupt:
        model.stop_training = True
    except Exception as e:
        shutil.rmtree(out_dir)
        no_exception = False
        raise e
    finally:
        configs.remove(config, _from=".running")
        if no_exception:
            configs.add(config, to=".archive")
            model_path = os.path.join(out_dir, "final_model.h5")
            print("\nSaving {}".format(model_path))
            model.keras.save(model_path)
        if gpu_queue is not None:
            gpu_queue.put(gpu_idx)

    return model.keras


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a fiber tracking model")

    parser.add_argument("config_path", type=str, nargs="?",
        help="Path to model config.")

    parser.add_argument("--model_name", type=str, choices=list(MODELS.keys()),
        help="Name of model to be trained.")

    parser.add_argument("--model_type", type=str,
        choices=["prior", "conditional"],
        help="Specify if model has type conditional or prior.")

    parser.add_argument("--train_path", type=str,
        help="Path to training samples.")

    parser.add_argument("--eval", type=str, dest="eval_path",
        help="Path to evaluation samples.")

    parser.add_argument("--epochs", type=int, help="Number of training epochs")

    parser.add_argument("--batch_size", type=int, help="batch size")

    parser.add_argument("--opt", type=str, dest="optimizer",
        help="Optimizer name.")

    parser.add_argument("--lr", type=float, dest="learning_rate",
                        help="Learning rate.")

    args, more_args = parser.parse_known_args()

    config = configs.compile_from(args.config_path, args, more_args)

    configs.check(config)

    train(config)