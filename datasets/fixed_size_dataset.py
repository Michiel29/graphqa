import numpy.random as rd
import numpy as np

from fairseq.data import BaseWrapperDataset


class FixedSizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, size=-1, seed=0):
        super().__init__(dataset)
        if size > 0:
            old_rd_state = rd.get_state()
            rd.seed(seed)
            self.data_indices = rd.choice(len(dataset), size, replace=False)
            rd.set_state(old_rd_state)
        else:
            self.data_indices = range(len(dataset))

        self._sizes = self.dataset.sizes[self.data_indices]

    @property
    def sizes(self):
        return self._sizes

    def __getitem__(self, index):
        return self.dataset[self.data_indices[index]]

    def __len__(self):
        return len(self.data_indices)

    def num_tokens(self, index):
        return self.dataset.num_tokens(self.data_indices[index])

    def size(self, index):
        return self.dataset.size(self.data_indices[index])
