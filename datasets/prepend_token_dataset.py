import numpy as np
import torch

from fairseq.data import BaseWrapperDataset


class PrependTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, token, keys):
        super().__init__(dataset)
        self.token = token
        if not isinstance(keys, list):
            self.keys = [keys]
        else:
            self.keys = keys
        self._sizes = None

    def __getitem__(self, idx):
        item = self.dataset[idx]
        ntokens = item.get('ntokens', 0)

        for key in self.keys:
            if isinstance(item[key], list):
                for i in range(len(item[key])):
                    item[key][i] = torch.cat([item[key][i].new([self.token]), item[key][i]])
                    ntokens += 1
            else:
                item[key] = torch.cat([item[key].new([self.token]), item[key]])
                ntokens += 1

        if 'ntokens' in item:
            item['ntokens'] = ntokens
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