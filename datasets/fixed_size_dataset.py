import numpy.random as rd
import numpy as np

from fairseq.data import BaseWrapperDataset


class FixedSizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, size=-1):
        super().__init__(dataset)
        if size > 0:
            self.data_indices = rd.choice(len(dataset), size, replace=False)
        else:
            self.data_indices = range(len(dataset))

        self._sizes = [self.size(idx) for idx in range(len(self.data_indices))]

    def __getitem__(self, index):
        return self.dataset[self.data_indices[index]]

    def __len__(self):
        return len(self.data_indices)

    def num_tokens(self, index):
        return self.dataset.num_tokens(self.data_indices[index])

    def size(self, index):
        return self.dataset.size(self.data_indices[index])

    @property
    def sizes(self):
        return self._sizes

    def ordered_indices(self):
        """Sorts by sentence length, randomly shuffled within sentences of """
        order = np.arange(len(self.data_indices))
        np.random.shuffle(order)
        order = [order]
        order.append(self.sizes)
        indices = np.lexsort(order)

        return indices