import logging
import math
import numpy as np
import time

from fairseq.data import BaseWrapperDataset
from utils.data_utils import numpy_seed
from utils.plasma_utils import maybe_move_to_plasma


logger = logging.getLogger(__name__)


class EpochSplitDataset(BaseWrapperDataset):
    """Splits dataset into several epochs with approximately epoch_size samples."""
    def __init__(self, dataset, epoch_size, seed):
        super().__init__(dataset)
        self.epoch_size = epoch_size
        self.seed = seed
        assert self.epoch_size is not None and self.epoch_size > 0
        self.epoch_splits = None
        self.epoch = None

    def set_dataset_epoch(self, dataset_epoch):
        if dataset_epoch != self.dataset.epoch:
            self.dataset.set_epoch(dataset_epoch)

    def set_epoch(self, epoch):
        if epoch == self.epoch:
            return

        assert epoch >= 1
        self.epoch = epoch

        if self.epoch_splits is None:
            self.set_dataset_epoch(1)
            self.epoch_splits = math.ceil(len(self.dataset) / self.epoch_size)
            assert self.epoch_splits >= 1
            logger.info('set epoch_split to be %d given epoch_size=%d, dataset size=%d' % (
                self.epoch_splits,
                len(self.dataset),
                self.epoch_size
            ))

        dataset_epoch = ((epoch - 1) // self.epoch_splits) + 1
        epoch_offset = (epoch - 1) % self.epoch_splits

        self.set_dataset_epoch(dataset_epoch)

        start_time = time.time()
        data_per_epoch = len(self.dataset) // (self.epoch_splits or 1)
        data_start = data_per_epoch * epoch_offset
        data_end = min(len(self.dataset), data_per_epoch * (epoch_offset + 1))

        with numpy_seed('EpochSplitDataset', self.seed, self.dataset.epoch):
            dataset_indices = np.random.permutation(len(self.dataset))
        self._indices = dataset_indices[data_start:data_end]
        self._sizes = self.dataset.sizes[self._indices]

        self._indices = maybe_move_to_plasma(self._indices)
        self._sizes = maybe_move_to_plasma(self._sizes)

        logger.info('selected %d samples from generation epoch %d and epoch offset %d in %d seconds' % (
            data_end - data_start,
            self.dataset.epoch,
            epoch_offset,
            time.time() - start_time,
        ))

    def __getitem__(self, index):
        return self.dataset[self._indices.array[index]]

    def __len__(self):
        return len(self._indices.array)

    def num_tokens(self, index):
        return self._sizes.array[index]

    def size(self, index):
        return self._sizes.array[index]

    @property
    def sizes(self):
        return self._sizes.array

    def ordered_indices(self):
        return np.argsort([10 * (np.random.random(len(self.sizes)) - 0.5) + self.sizes])[0]
