from os import listdir
from os.path import isfile, join, isdir
import numpy as np

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from dipy.io.gradients import read_bvals_bvecs


class Samples(Sequence):
    def __init__(self, config):
        """"""
        self.batch_size = config['batch_size']
        self.istraining = config['istraining']
        if isinstance(config['sample_path'], list) and \
                not isdir(config['sample_path'][0]):
            if isinstance(self, RNNSamples):
                raise NotImplementedError("Do RNNSamples support several subjects?")
            self.sample_files = [np.load(p, allow_pickle=True)
                for p in config['sample_path']]
            self.n_samples = np.sum([s["n_samples"] for s in self.sample_files])

            self.samples = {}
            np.random.seed(42)
            perm = np.random.permutation(self.n_samples)
            print("Merging data from several subjects ...")
            for key in self.sample_files[0].files:
                self.samples[key] = []
                for f in self.sample_files:
                    if f[key].ndim < 2:
                        self.samples[key].append(np.expand_dims(f[key], -1))
                    else:
                        self.samples[key].append(f[key])
                self.samples[key] = np.vstack(self.samples[key])
                if self.samples[key].shape[0] == self.n_samples:
                    self.samples[key] = self.samples[key][perm]
        elif isinstance(config['sample_path'], list) and \
                isdir(config['sample_path'][0]):
            self.samples = {}
            self.sample_files = [join(subject, f)
                                 for subject in config['sample_path']
                                 for f in listdir(subject)
                                 if isfile(join(subject, f))
                                 and 'samples' in f]

            self.sample_shapes = []
            for sample in self.sample_files:
                sample_i = np.load(sample, allow_pickle=True)
                sample_i_shape = sample_i['sample_shape']
                self.n_samples += sample_i["n_samples"]
                self.sample_shapes.append(sample_i_shape)

        elif isdir(config['sample_path']):
            self.samples = {}
            self.sample_files = sorted([join(config['sample_path'], f)
                                        for f in listdir(config['sample_path'])
                                        if isfile(join(config['sample_path'], f))
                                        and 'samples' in f])
            self.sample_shapes = []
            for sample in self.sample_files:
                sample_i = np.load(sample, allow_pickle=True)
                sample_i_shape = sample_i['sample_shape']
                self.n_samples = sample_i["n_samples"]
                self.sample_shapes.append(sample_i_shape)
        else:
            self.samples = np.load(config['sample_path'], allow_pickle=True)  # lazy loading
            self.n_samples = self.samples["n_samples"]

    def __len__(self):
        if 'inputs' in self.samples:
            if self.istraining:
                return self.n_samples // self.batch_size  # drop remainder
            else:
                return np.ceil(self.n_samples / self.batch_size).astype(int)
        else:
            return self.batch_indices[-1]


class FvMSamples(Samples):

    def __init__(self, *args, **kwargs):
        super(FvMSamples, self).__init__(*args, **kwargs)
        print("Loading {} samples...".format("train" if self.istraining else "eval"))
        if 'inputs' in self.samples and 'outgoing' in self.samples:
            self.current_idx = None
            self.inputs = self.samples["inputs"]
            self.outgoing = self.samples["outgoing"]
        else:
            self.current_idx = -1
            self.inputs = None
            self.outgoing = None

            # Cut the data to fir the batch size
            self.new_shapes = self.inputs = \
                [(batch_shape[0] - (batch_shape[0] % self.batch_size), batch_shape[1])
                 if batch_shape[0] > self.batch_size else (batch_shape[0], batch_shape[1])
                 for batch_shape in self.sample_shapes]

            # Remove files that have not enough data for a batch
            self.sample_files = [self.sample_files[i] for i, shape in
                                 enumerate(self.new_shapes) if
                                 shape[0] >= self.batch_size]
            self.new_shapes = [shape for shape in self.new_shapes if
                               shape[0] >= self.batch_size]

            # To help find the right fiber for the right batch index
            self.batch_indices = np.cumsum(
                [(shape[0] // self.batch_size) for shape in self.new_shapes])

            if len(self.new_shapes) < 1:
                raise Exception('Data input is empty. This can happen when '
                                'number of same length fibers are less than the '
                                'selected batch size. Maybe reduce the batch size'
                                'and check again!')

            print(
                "Reduced length of input from {0} to {1} to fit the batch size.".
                format(len(self.sample_shapes), len(self.new_shapes)))

    def __getitem__(self, idx):
        if self.current_idx is not None:
            # Case of several files:

            first_index = np.where((idx < self.batch_indices))[0][0]
            if first_index != self.current_idx:
                # If batch is not in the cached array:

                # Load the correct file from the path
                current_path = self.sample_files[first_index]
                current_shape = self.new_shapes[first_index]
                samples = np.load(current_path, allow_pickle=True)
                first_possible_input = samples['inputs']
                first_possible_output = samples['outgoing']

                # Cut the data to fir the batch
                first_possible_input = first_possible_input[:current_shape[0],
                                       ...]
                first_possible_output = first_possible_output[:current_shape[0],
                                        ...]

                # Update the cache
                self.inputs = first_possible_input
                self.outgoing = first_possible_output
                self.current_idx = first_index

            previous_index = first_index - 1
            if previous_index < 0:
                previous_index = 0
            else:
                previous_index = self.batch_indices[previous_index]
            idx = idx - previous_index

        x_batch = self.inputs[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch = self.outgoing[idx * self.batch_size:(idx + 1) * self.batch_size]
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
        self.current_idx = -1
        self.current_input = None
        self.current_output = None

        self.new_shapes = self.inputs = [
            (batch_shape[0] - (batch_shape[0] % self.batch_size),
             batch_shape[1], batch_shape[2])
            if batch_shape[0] > self.batch_size else
            (batch_shape[0], batch_shape[1], batch_shape[2])
            for batch_shape in self.sample_shapes]

        self.sample_files = [self.sample_files[i] for i, shape in
                             enumerate(self.new_shapes) if
                             shape[0] >= self.batch_size]
        self.new_shapes = [shape for shape in self.new_shapes if shape[0] >= self.batch_size]

        # To help find the right fiber for the right batch index
        self.batch_indices = np.cumsum([(shape[0] // self.batch_size) * shape[1] for shape in self.new_shapes])

        if len(self.new_shapes) < 1:
            raise Exception('Data input is empty. This can happen when '
                            'number of same length fibers are less than the '
                            'selected batch size. Maybe reduce the batch size'
                            'and check again!')

        print("Reduced length of input from {0} to {1} to fit the batch size.".
              format(len(self.sample_shapes), len(self.new_shapes)))

        self.reset_batches = self._get_reset_batches()

    def __len__(self):
        return self.batch_indices[-1]

    def __getitem__(self, idx):
        first_index = np.where((idx < self.batch_indices))[0][0]
        if first_index == self.current_idx:
            first_possible_input = self.current_input
            first_possible_output = self.current_output
        else:
            # Load the correct file from the path
            current_path = self.sample_files[first_index]
            current_shape = self.new_shapes[first_index]
            samples = np.load(current_path, allow_pickle=True)
            first_possible_input = samples['inputs']
            first_possible_output = samples['outgoing']

            # Cut the data to fir the batch
            first_possible_input = first_possible_input[:current_shape[0], ...]
            first_possible_output = first_possible_output[:current_shape[0], ...]

            # Update the cache
            self.current_input = first_possible_input
            self.current_output = first_possible_output
            self.current_idx = first_index

        previous_index = first_index - 1
        if previous_index < 0:
            previous_index = 0
        else:
            previous_index = self.batch_indices[previous_index]
        current_batch_idx = idx - previous_index
        row_idx = current_batch_idx // (first_possible_input.shape[1])
        col_idx = current_batch_idx % first_possible_input.shape[1]

        x_batch = first_possible_input[row_idx * self.batch_size:(row_idx + 1)*self.batch_size, col_idx, np.newaxis, ...]
        y_batch = first_possible_output[row_idx * self.batch_size:(row_idx + 1)*self.batch_size, col_idx, np.newaxis, ...]

        return x_batch, {"fvm": y_batch, "kappa": np.zeros(len(y_batch))}

    def _get_reset_batches(self):
        """For sure there is a smarter way for this ... :)"""
        reset_batches = []
        for idx in range(self.__len__()):
            first_index = np.where((idx < self.batch_indices))[0][0]
            first_possible_shape = self.new_shapes[first_index]
            previous_index = first_index - 1

            if previous_index < 0:
                previous_index = 0
            else:
                previous_index = self.batch_indices[previous_index]
            current_batch_idx = idx - previous_index
            col_idx = current_batch_idx % first_possible_shape[1]
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


class ClassifierSamples(FvMSamples):

    def __init__(self, *args, **kwargs):
        super(ClassifierSamples, self).__init__(*args, **kwargs)
        print("Loading {} samples...".format("train" if self.istraining else "eval"))
        configs = args[0]
        _, self.bvecs = read_bvals_bvecs(None, configs["bvec_path"])

    def __getitem__(self, idx):
        inputs, outgoing = super(ClassifierSamples, self).__getitem__(idx)
        out_classes = to_categorical(
            np.array([np.argmax([np.dot(base_vec, outvec)
                                  for base_vec in self.bvecs])
                       for outvec in outgoing]), num_classes=len(self.bvecs))
        return inputs, out_classes