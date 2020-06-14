from collections import defaultdict
from itertools import product, chain
import numpy as np
import numpy.random as rd
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import data_utils, FairseqDataset

from datasets import AnnotatedText
from utils.probing_utils import tacred_relations, tacred_rules

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
        self.all_rules = list(product(range(n_relations), repeat=3))
        for rule in tacred_rules:
            self.all_rules = list(filter((rule).__ne__, self.all_rules))
        self.rule_indices = np.random.choice(len(self.all_rules), size=n_rules-len(tacred_rules), replace=False) + len(tacred_rules)

        self.all_rules = tacred_rules + self.all_rules
        self.rule_indices = np.concatenate((np.array(range(len(tacred_rules))), self.rule_indices))      


    def __getitem__(self, index):

        rule = self.all_rules[self.rule_indices[index]]

        # from utils.diagnostic_utils import Diagnostic
        # diag = Diagnostic(self.dictionary, entity_dictionary=None)
        # tmp = diag.decode_text(self.tacred_dataset.__getitem__(self.relation_index[0][0])['text'])

        graph_list = [[] for x in range(self.n_texts)]
        target_list = []
        all_text_indices = []
        for rel in rule:
            cur_text_indices = list(rd.choice(self.relation_index[rel], size=self.n_texts, replace=True))
            all_text_indices += cur_text_indices
        all_text_indices = np.array(all_text_indices)

        unique_text_indices = np.unique(all_text_indices, return_counts=False)
        unique_text_lengths = np.array([len(self.tacred_dataset.__getitem__(x)['text']) for x in unique_text_indices])
        text_indices = np.argsort(unique_text_lengths)
        text = [self.tacred_dataset.__getitem__(x)['text'] for x in text_indices]
        
        all_text_indices = all_text_indices.reshape(3, -1)
        for i in range(len(all_text_indices)):
            cur_text_indices = all_text_indices[i]
            for j, idx in enumerate(cur_text_indices):
                new_idx = text_indices[np.where(unique_text_indices == idx)[0][0]]
                if i == 0:
                    target_list.append(new_idx)
                else:
                    graph_list[j].append(new_idx)

        item = {
            'text': text,
            'target_text_idx': target_list,
            'graph': graph_list,
            'graph_sizes': [1] * self.n_texts,
            'target_relation': tacred_relations[rule[0]],
            'evidence_relations': (tacred_relations[rule[1]], tacred_relations[rule[2]])
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
        assert len(instances) == 1

        text = instances[0]['text']
        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())
        ntokens = padded_text.numel()
        nsentences = len(padded_text)

        batch = {
            'text': padded_text.unsqueeze(0),
            'target_text_idx': torch.LongTensor(instances[0]['target_text_idx']),
            'graph': torch.LongTensor(instances[0]['graph']),
            'graph_sizes': torch.LongTensor(instances[0]['graph_sizes']),
            'target_relation': instances[0]['target_relation'],
            'evidence_relations': instances[0]['evidence_relations'],
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }

        return batch

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
