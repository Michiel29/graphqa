import numpy.random as rd
import numpy as np

from fairseq.data import BaseWrapperDataset, data_utils


class FilteredDataset(BaseWrapperDataset):
    def __init__(self, dataset, data_indices):
        super().__init__(dataset)
        self.data_indices = data_indices
        self._sizes = self.dataset.sizes[self.data_indices]

    @property
    def sizes(self):
        return self._sizes

    def __getitem__(self, index, **kwargs):
        return self.dataset.__getitem__(self.data_indices[index], **kwargs)

    def __len__(self):
        return len(self.data_indices)

    def num_tokens(self, index):
        return self.dataset.num_tokens(self.data_indices[index])

    def size(self, index):
        return self.dataset.size(self.data_indices[index])

    def ordered_indices(self):
        """Sorts by sentence length, randomly shuffled within sentences of same length"""
        return np.lexsort([
            np.random.permutation(len(self)),
            self.sizes,
        ])


def filter_by_max_length(dataset, max_positions, size=-1):
    data_indices = np.nonzero(dataset.sizes <= max_positions)[0]
    return FilteredDataset(dataset, data_indices), data_indices


def prune_dataset_size(dataset, size, seed):
    if size == -1:
        return dataset
    else:
        with data_utils.numpy_seed(hash('prune_dataset_size'), seed):
            data_indices = rd.choice(len(dataset), size, replace=False)
        return FilteredDataset(dataset, data_indices)