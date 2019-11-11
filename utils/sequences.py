import numpy as np

from tensorflow.keras.utils import Sequence


class Samples(Sequence):
    def __init__(self,
                 sample_path,
                 batch_size=256,
                 istraining=True,
                 max_n_samples=np.inf):
        """"""
        self.batch_size = batch_size
        self.istraining = istraining
        if isinstance(sample_path, list):
            if isinstance(self, RNNSamples):
                raise NotImplementedError("Do RNNSamples support several subjects?")
            self.sample_files = [np.load(p, allow_pickle=True)
                for p in sample_path]
            self.n_samples = np.sum([s["n_samples"] for s in self.sample_files])

            self.samples = {}
            np.random.seed(42)
            perm = np.random.permutation(self.n_samples)
            for key in self.sample_files[0].files:
                data = []
                for f in self.sample_files:
                    if f[key].ndim < 2:
                        data.append(np.expand_dims(f[key], -1))
                    else:
                        data.append(f[key])
                self.samples[key] = np.vstack(data)
                if self.samples[key].shape[0] == self.n_samples:
                    self.samples[key] = self.samples[key][perm]
        else:
            self.samples = np.load(sample_path, allow_pickle=True)  # lazy loading
            self.n_samples = self.samples["n_samples"]

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
            self.inputs = [batch_input[:-(batch_input.shape[0] % self.batch_size),...]
                                    if batch_input.shape[0] > self.batch_size else batch_input
                                    for batch_input in self.samples["inputs"] ]
            self.outgoing = [batch_input[:-(batch_input.shape[0] % self.batch_size), ...]
                                      if batch_input.shape[0] > self.batch_size else batch_input
                                      for batch_input in self.samples["outgoing"]]
        self.inputs = [batch_input for batch_input in self.inputs if batch_input.shape[0] >= self.batch_size]
        self.outgoing = [batch_input for batch_input in self.outgoing if batch_input.shape[0] >= self.batch_size]

        print("Reduced length of input from {0} to {1} to fit the batch size.".
              format(len(self.samples["inputs"]), len(self.inputs)))

        # To help find the right fiber for the right batch index
        self.batch_indices = np.cumsum([(fiber.shape[0] // self.batch_size) * fiber.shape[1] for fiber in self.inputs])
        self.reset_batches = self._get_reset_batches()

    def __len__(self):
        return self.batch_indices[-1]

    def __getitem__(self, idx):
        first_index = np.where((idx < self.batch_indices))[0][0]
        first_possible_input = self.inputs[first_index]
        first_possible_output = self.outgoing[first_index]
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

        return (x_batch, y_batch)

    def _get_reset_batches(self):
        """For sure there is a smarter way for this ... :)"""
        reset_batches = []
        for idx in range(self.__len__()):
            first_index = np.where((idx < self.batch_indices))[0][0]
            first_possible_input = self.inputs[first_index]
            previous_index = first_index - 1

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