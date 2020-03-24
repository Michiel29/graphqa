from collections import defaultdict
import numpy as np
import numpy.random as rd
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import data_utils, FairseqDataset

from datasets import AnnotatedTextDataset


class KBP37Dataset(FairseqDataset):

    def __init__(
        self,
        split,
        annotation_text_dataset,
        relation_dataset,
        dictionary,
        entity_dictionary,
        mask_type,
        seed,
    ):
        self.split = split
        self.annotation_text_dataset = annotation_text_dataset
        self.relation_dataset = relation_dataset
        self.dictionary = dictionary
        self.entity_dictionary = entity_dictionary
        self.seed = seed
        self.epoch = 0

        self.relation_index = defaultdict(list)
        for idx in range(len(self.relation_dataset)):
            self.relation_index[self.relation_dataset[idx].item()].append(idx)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed(271828, self.seed, self.epoch, index):
            annot_item = self.annotation_text_dataset[index]
            relation = self.relation_dataset[index]

        item = {
            'text': annot_item['text'],
            'target': relation
        }

        return item

    def __len__(self):
        return len(self.annotation_text_dataset)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @property
    def sizes(self):
        return self.annotation_text_dataset.sizes

    def ordered_indices(self):
        """Sorts by sentence length, randomly shuffled within sentences of same length"""
        return np.lexsort([
            np.random.permutation(len(self)),
            self.sizes,
        ])

    def collater(self, instances):
        batch_size = len(instances)

        if batch_size == 0:
            return None

        text, target = [], []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            target.append(instance['target'])
            ntokens += len(instance['text'])
            nsentences += 1

        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())

        batch = {
            'split': self.split,
            'text': padded_text,
            'target': torch.LongTensor(target),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }

        return batch

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
