import numpy as np
import torch

from fairseq.data import BaseWrapperDataset


class PrependTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, token, keys=None):
        super().__init__(dataset)
        self.token = token
        if not isinstance(keys, list):
            self.keys = [keys]
        else:
            self.keys = keys
        self._sizes = None

    def __getitem__(self, idx):
        item = self.dataset[idx]
        ntokens = item.get['ntokens'] if hasattr(item, 'ntokens') else 0

        for key in self.keys:
            if isinstance(item[key], list):
                for i in range(len(item[key])):
                    item[key][i] = torch.cat([item[key][i].new([self.token]), item[key][i]])
                    ntokens += 1
            elif key is not None:
                item[key] = torch.cat([item[key].new([self.token]), item[key]])
                ntokens += 1
            else:
                item = torch.cat([item.new([self.token]), item])
                ntokens += 1

        if hasattr(item, 'ntokens'):
            item['ntokens'] = ntokens
        return item

    @property
    def sizes(self):
        return self.dataset.sizes + 1

    def num_tokens(self, index):
        return self.dataset.num_tokens(index) + 1

    def size(self, index):
        return self.dataset.size(index) + 1