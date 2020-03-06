from collections import defaultdict
import logging
import numpy as np
import numpy.random as rd
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import data_utils, FairseqDataset

from datasets import AnnotatedTextDataset

logger = logging.getLogger(__name__)


class FewRelDataset(FairseqDataset):

    def __init__(
        self,
        annotation_text_dataset,
        relation_dataset,
        dictionary,
        mask_type,
        n_way,
        n_shot,
        seed,
    ):
        self.annotation_text_dataset = annotation_text_dataset
        self.relation_dataset = relation_dataset
        self.dictionary = dictionary
        self.n_way = n_way
        assert self.n_way > 1
        self.n_shot = n_shot
        assert self.n_way > 0
        self.seed = seed
        self.epoch = 0

        self.relation_index = defaultdict(list)
        for idx in range(len(self.relation_dataset)):
            self.relation_index[self.relation_dataset[idx].item()].append(idx)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed(hash(self.__class__), self.seed, self.epoch, index):
            target_item = self.annotation_text_dataset[index]
            target_relation = self.relation_dataset[index]
            relations = rd.choice(
                list(self.relation_index.keys()),
                size=self.n_way,
                replace=False,
            ).tolist()
            if target_relation in relations:
                relations.remove(target_relation)
            else:
                relations = relations[:self.n_way - 1]

            relations = [target_relation.item()] + relations

            exemplars = []
            for rel in relations:
                rel_examplar_idxs = rd.choice(self.relation_index[rel], size=self.n_shot, replace=False)
                exemplars += [
                    self.annotation_text_dataset[idx]['text']
                    for idx in rel_examplar_idxs
                ]

            ntokens, nsentences = len(target_item['text']), 1
            for exemplar in exemplars:
                nsentences += 1
                ntokens += len(exemplars)

        return {
            'text': target_item['text'],
            'exemplars': exemplars,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }

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

        text = []
        exemplars = []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            exemplars += instance['exemplars']
            ntokens += len(instance['text']) + sum([len(s) for s in instance['exemplars']])
            nsentences += 1 + len(instance['exemplars'])

        padded_mention = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())
        padded_exemplars = pad_sequence(exemplars, batch_first=True, padding_value=self.dictionary.pad())

        return {
            'text': padded_mention,
            'exemplars': padded_exemplars,
            'target': torch.zeros(len(instances), dtype=torch.long),
            'batch_size': len(instances),
            'ntokens': ntokens,
            'nsentences': nsentences,
        }

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
