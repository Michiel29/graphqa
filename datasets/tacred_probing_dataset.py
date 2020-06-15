from collections import defaultdict
from itertools import product, chain
import numpy as np
import numpy.random as rd
from tqdm import tqdm

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
        n_strong_negs,
        dictionary,
        seed,
    ):
        self.tacred_dataset = tacred_dataset
        self.n_rules = n_rules
        self.n_texts = n_texts
        self.n_strong_negs = n_strong_negs
        self.dictionary = dictionary
        self.seed = seed
        
        self.relation_dataset = tacred_dataset.relation_dataset
        self.relation_index = defaultdict(list)
        for idx in range(len(self.relation_dataset)):
            self.relation_index[self.relation_dataset[idx].item()].append(idx)
        n_relations = len(self.relation_index)

        # Create strong negative rules
        strong_neg_rules = []
        for rule in tqdm(tacred_rules, desc='Creating strong negative rules'):
            target, evidence_1, evidence_2 = rule
            cur_negs = []
            neg_candidates = np.random.choice(n_relations, size=n_relations, replace=False)
            for c in neg_candidates:
                if c == target:
                    continue
                candidate_rule = (c, evidence_1, evidence_2)
                if candidate_rule not in tacred_rules + strong_neg_rules:
                    cur_negs.append(candidate_rule)
                if len(cur_negs) == n_strong_negs:
                    break
            strong_neg_rules += cur_negs
            
        # Create weak negative rules
        self.all_rules = list(product(range(n_relations), repeat=3))
        n_non_weak_rules = len(tacred_rules) + len(strong_neg_rules)
        for rule in tqdm(tacred_rules + strong_neg_rules, desc='Creating weak negative rules'):
            self.all_rules = list(filter((rule).__ne__, self.all_rules))
        self.rule_indices = np.random.choice(len(self.all_rules), size=n_rules-n_non_weak_rules, replace=False) + n_non_weak_rules

        # Combine all rules together
        self.all_rules = tacred_rules + strong_neg_rules + self.all_rules
        self.rule_indices = np.concatenate((np.array(range(n_non_weak_rules)), self.rule_indices))     


    def __getitem__(self, index):

        rule = self.all_rules[self.rule_indices[index]]

        graph_list = [[] for x in range(self.n_texts)]
        target_list = []
        all_text_indices = []
        for rel in rule:
            cur_text_indices = list(rd.choice(self.relation_index[rel], size=self.n_texts, replace=True))
            all_text_indices += cur_text_indices
        text = list(set(all_text_indices))
        all_text_indices = np.array(all_text_indices)
        all_text_indices = all_text_indices.reshape(3, -1)

        for i in range(len(all_text_indices)):
            cur_text_indices = all_text_indices[i]
            for j, idx in enumerate(cur_text_indices):
                if i == 0:
                    target_list.append(idx)
                else:
                    graph_list[j].append(idx)

        item = {
            'text': text,
            'target_text_idx': target_list,
            'graph': graph_list,
            'graph_sizes': [1] * self.n_texts,
            'target_relation': [tacred_relations[rule[0]]],
            'evidence_relations': [tacred_relations[rule[1]], tacred_relations[rule[2]]]
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

        all_text_indices, target_text_idx, graph, graph_sizes = [], [], [], []
        target_relation, evidence_relations = [], []
        ntokens, nsentences = 0, 0

        # Get text passages and indices
        for instance in instances:
            all_text_indices += instance['text']
        unique_text_indices = list(set(all_text_indices))
        text_lengths = np.array([len(self.tacred_dataset.__getitem__(x)['text']) for x in unique_text_indices])
        argsort_text_indices = np.argsort(text_lengths)
        text = [self.tacred_dataset.__getitem__(unique_text_indices[x])['text'] for x in argsort_text_indices]
        text_indices = np.array(unique_text_indices)[argsort_text_indices]

        # Create text clusters
        sorted_text_lengths = np.sort(text_lengths)
        text_mean = np.mean(sorted_text_lengths)
        text_std = np.std(sorted_text_lengths)
        bin_vals = [0] + [text_mean + 0.5*k*text_std for k in range(-3, 4)] + [float('inf')]
        text_cluster_indices = [np.where(np.logical_and(sorted_text_lengths > bin_vals[i], sorted_text_lengths <= bin_vals[i+1]))[0] for i in range(len(bin_vals)-1)]

        # Pad each text cluster
        text_clusters = []
        for c in text_cluster_indices:
            cur_cluster = [text[x] for x in c]
            padded_text = pad_sequence(cur_cluster, batch_first=True, padding_value=self.dictionary.pad())
            text_clusters.append(padded_text)
            ntokens += padded_text.numel()
            nsentences += len(padded_text)

        # Convert text indices
        for instance in instances:
            target_text_idx += [np.where(text_indices == x)[0][0] for x in instance['target_text_idx']]
            graph += [[np.where(text_indices == x1)[0][0], np.where(text_indices == x2)[0][0]] for (x1, x2) in instance['graph']]
            graph_sizes += instance['graph_sizes']
            target_relation += instance['target_relation']
            evidence_relations += [instance['evidence_relations']]

        batch = {
            'text': text_clusters,
            'target_text_idx': torch.LongTensor(target_text_idx),
            'graph': torch.LongTensor(graph),
            'graph_sizes': torch.LongTensor(graph_sizes),
            'target_relation': target_relation,
            'evidence_relations': evidence_relations,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }

        return batch

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
