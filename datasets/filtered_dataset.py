import logging
import numpy as np

from fairseq.data import BaseWrapperDataset, data_utils
from utils.data_utils import numpy_seed

logger = logging.getLogger(__name__)


class FilteredDataset(BaseWrapperDataset):
    def __init__(self, dataset, data_indices_fn):
        super().__init__(dataset)
        self.data_indices_fn = data_indices_fn

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
        self.data_indices = self.data_indices_fn(self.dataset)
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
        return self._sizes[index]

    def ordered_indices(self):
        return np.argsort([10 * (np.random.random(len(self._sizes)) - 0.5) + self._sizes])[0]


class FixedSizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, size, seed):
        super().__init__(dataset)
        self._size = size
        self.seed = seed

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
        with numpy_seed('FixedSizeDataset', self.seed):
            if len(self.dataset) <= size:
                self.data_indices = np.arange(len(self.dataset))
            else:
                self.data_indices = np.random.choice(
                    len(self.dataset),
                    self._size,
                    replace=False,
                )
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
        return self._sizes[index]

    def ordered_indices(self):
        return np.argsort([10 * (np.random.random(len(self._sizes)) - 0.5) + self._sizes])[0]
