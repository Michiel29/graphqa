from collections import defaultdict
from itertools import permutations, chain
import numpy as np
import numpy.random as rd
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import data_utils, FairseqDataset

from datasets import AnnotatedText


class TACREDProbingDataset(FairseqDataset):

    def __init__(
        self,
        tacred_dataset,
        n_rules,
        n_texts,
        dictionary,
        seed,
    ):
        self.tacred_dataset = tacred_dataset
        self.n_rules = n_rules
        self.n_texts = n_texts
        self.dictionary = dictionary
        self.seed = seed
        
        self.relation_dataset = tacred_dataset.relation_dataset
        self.relation_index = defaultdict(list)
        for idx in range(len(self.relation_dataset)):
            self.relation_index[self.relation_dataset[idx].item()].append(idx)
        
        n_relations = len(self.relation_index)
        self.perm = list(permutations(range(n_relations), 3))
        self.perm_indices = np.random.choice(len(self.perm), size=n_rules, replace=False)


    def __getitem__(self, index):

        rule = self.perm[self.perm_indices[index]]

        graph_list = [[] for x in range(self.n_texts)]
        for rel in rule:
            text_indices = rd.choice(self.relation_index[rel], size=self.n_texts, replace=False)
            for i, text_index in enumerate(text_indices):
                cur_text = self.tacred_dataset.__getitem__(text_index)['text']
                graph_list[i].append(cur_text)

        graph_list = list(chain(*graph_list))

        item = {
            'text': graph_list,
        }

        return item

    def __len__(self):
        return self.n_rules

    def num_tokens(self, index):
        return 1

    def size(self, index):
        return 1

    @property
    def sizes(self):
        return np.ones(len(self.n_rules))

    # def ordered_indices(self):
    #     return np.argsort([10 * (np.random.random(len(self.sizes)) - 0.5) + self.sizes])[0]

    def collater(self, instances):
        batch_size = len(instances)

        if batch_size == 0:
            return None

        text = []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text += instance['text']
            ntokens += len(instance['text'])
            nsentences += 1

        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())

        batch = {
            'text': padded_text,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }

        return batch

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
