import numpy as np

from fairseq.data import BaseWrapperDataset


class ShuffledDataset(BaseWrapperDataset):

    def __init__(self, dataset, sizes):
        super().__init__(dataset)
        self._sizes = sizes

    @property
    def sizes(self):
        return self._sizes

    def ordered_indices(self):
        return np.argsort([10 * (np.random.random(len(self.sizes)) - 0.5) + self.sizes])[0]
