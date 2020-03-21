import numpy as np
import torch

from fairseq.data import BaseWrapperDataset


class PrependTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, token, key):
        super().__init__(dataset)
        self.token = token
        self.key = key
        self._sizes = None

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item[self.key] = torch.cat([item[self.key].new([self.token]), item[self.key]])
        return item

    @property
    def sizes(self):
        if self._sizes is None:
            self._sizes = np.array(dataset.sizes) + 1
        return self._sizes

    def num_tokens(self, index):
        return self.dataset.num_tokens(index) + 1

    def size(self, index):
        return self.dataset.size(index) + 1