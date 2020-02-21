import numpy.random as rd
import numpy as np

from fairseq.data import BaseWrapperDataset


class SelectDictionaryDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        super().__init__(dataset)
        self.key = key

    @property
    def sizes(self):
        return self.dataset.sizes

    def __getitem__(self, index):
        return self.dataset[index][self.key]
