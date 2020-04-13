from collections import OrderedDict
import numpy as np
from torch.utils.data.dataloader import default_collate

from fairseq.data import FairseqDataset


class DictionaryDataset(FairseqDataset):

    def __init__(self, dict_of_datasets, main_key):
        super().__init__()
        self.dict_of_datasets = dict_of_datasets
        self.main_key = main_key

    def set_epoch(self, epoch):
        for ds in self.dict_of_datasets.values():
            ds.set_epoch(epoch)

    def __getitem__(self, index):
        return OrderedDict((k, ds[index]) for k, ds in self.dict_of_datasets.items())

    def __len__(self):
        return len(self.dict_of_datasets[self.main_key])

    def num_tokens(self, index):
        return self.dict_of_datasets[self.main_key].num_tokens(index)

    def size(self, index):
        return self.dict_of_datasets[self.main_key].size(index)

    @property
    def sizes(self):
        return self.dict_of_datasets[self.main_key].sizes

    def collater(self, samples):
        if len(samples) == 0:
            return None
        sample = OrderedDict()
        for k, ds in self.dict_of_datasets.items():
            try:
                sample[k] = ds.collater([s[k] for s in samples])
            except NotImplementedError:
                sample[k] = default_collate([s[k] for s in samples])
        return sample

    def ordered_indices(self):
        return np.argsort([10 * (np.random.random(len(self.sizes)) - 0.5) + self.sizes])[0]

    @property
    def supports_prefetch(self):
        return False
