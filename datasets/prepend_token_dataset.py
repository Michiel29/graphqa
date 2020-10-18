import numpy as np
import torch

from fairseq.data import BaseWrapperDataset


class PrependTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, token, keys=None, annotation_keys=None):
        super().__init__(dataset)
        self.token = token
        if not isinstance(keys, list):
            self.keys = [keys]
        else:
            self.keys = keys
        self.annotation_keys = annotation_keys
        self._sizes = None


    def __getitem__(self, idx):
        item = self.dataset[idx]
        if item is None:
            return None
        ntokens = item.get['ntokens'] if hasattr(item, 'ntokens') else 0

        for key in self.keys:
            if isinstance(item[key], list):
                for i in range(len(item[key])):
                    current_item = item[key][i]
                    if current_item.ndim == 2:
                        t = current_item.new_full(size=(current_item.shape[0], 1), fill_value=self.token)
                        item[key][i] = torch.cat([t, current_item], dim=1)
                        ntokens += t.numel()
                    else:
                        item[key][i] = torch.cat([current_item.new([self.token]), current_item])
                        ntokens += 1
            elif key is not None:
                if item[key].ndim == 2:
                    t = item[key].new_full(size=(item[key].shape[0], 1), fill_value=self.token)
                    item[key] = torch.cat([t, item[key]], dim=1)
                    ntokens += t.numel()
                else:
                    item[key] = torch.cat([item[key].new([self.token]), item[key]])
                    ntokens += 1
            else:
                item = torch.cat([item.new([self.token]), item])
                ntokens += 1

        if self.annotation_keys is not None:
            for key in self.annotation_keys:
                if key in item and item[key] is not None:
                    item[key] = item[key] + 1

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