import copy
from functools import lru_cache
import numpy as np
import numpy.random as rd
import torch

from fairseq.data import FairseqDataset
from fairseq.data import data_utils

from datasets import AnnotatedText, GraphDataset


class RelInfDataset(FairseqDataset):

    def __init__(
        self,
        annotated_text,
        graph,
        k_negative,
        n_entities,
        seed,
        same_replace_heads_for_all_negatives,
        negative_split_probs,
        use_sentence_negatives,
    ):
        self.annotated_text = annotated_text
        self.use_sentence_negatives = use_sentence_negatives
        self.k_negative = k_negative
        if self.use_sentence_negatives:
            assert self.k_negative == 0
        self.n_entities = n_entities
        self.graph = graph
        self.seed = seed
        self.epoch = None
        self.same_replace_heads_for_all_negatives = same_replace_heads_for_all_negatives
        self.negative_split_probs = copy.deepcopy(negative_split_probs)
        assert len(self.negative_split_probs) == 3, self.negative_split_probs
        assert sum(self.negative_split_probs) <= 1, self.negative_split_probs
        self.negative_split_probs.append(1 - sum(self.negative_split_probs))

    def set_epoch(self, epoch):
        self.graph.set_epoch(epoch)
        self.epoch = epoch

    def sample_neighbors(self, neighbors_lists, n_samples_per_category):
        assert len(neighbors_lists) == 3
        assert len(n_samples_per_category) == 4
        result = []
        n_samples = 0
        for i, neighbors_list in enumerate(neighbors_lists):
            n_samples += n_samples_per_category[i]
            actually_sampled = min(n_samples, len(neighbors_list))
            if actually_sampled > 0:
                result.extend(rd.choice(neighbors_list, size=actually_sampled, replace=False))
            n_samples -= actually_sampled
        n_samples += n_samples_per_category[3]
        if n_samples > 0:
            result.extend(rd.randint(self.n_entities, size=n_samples))
        assert len(result) == sum(n_samples_per_category)
        return result

    @lru_cache(maxsize=8)
    def get_n_samples_per_category(self, num_replace_head):
        n_samples_per_category = [
            int(x * num_replace_head)
            for x in self.negative_split_probs
        ]
        while sum(n_samples_per_category) < num_replace_head:
            n_samples_per_category[rd.randint(4, size=1)[0]] += 1
        return n_samples_per_category

    def __getitem__(self, index):
        edge = self.graph[index]
        head = edge[GraphDataset.HEAD_ENTITY]
        tail = edge[GraphDataset.TAIL_ENTITY]
        item = {}

        with data_utils.numpy_seed(17101990, self.seed, self.epoch, index):
            item['text'] = self.annotated_text.annotate(*(edge.numpy()))
            item['nsentences'] = 1
            item['ntokens'] = len(item['text'])

            shall_replace_head = rd.randint(2, size=1)[0]
            if self.same_replace_heads_for_all_negatives:
                num_replace_head = shall_replace_head * self.k_negative
            else:
                num_replace_head = self.k_negative // 2 + shall_replace_head * (self.k_negative % 2)

            head_neighbors = self.graph.get_neighbors(head)
            tail_neighbors = self.graph.get_neighbors(tail)

            head_tail_neighbors = np.intersect1d(head_neighbors, tail_neighbors, assume_unique=True)
            head_only_neighbors = np.setdiff1d(head_neighbors, head_tail_neighbors, assume_unique=True)
            tail_only_neighbors = np.setdiff1d(tail_neighbors, head_tail_neighbors, assume_unique=True)

            heads = np.full(1 + self.k_negative, fill_value=head, dtype=np.int64)
            if num_replace_head > 0:
                heads[1: 1 + num_replace_head] = self.sample_neighbors(
                    [head_tail_neighbors, tail_only_neighbors, head_only_neighbors],
                    self.get_n_samples_per_category(num_replace_head),
                )
            item['head'] = heads

            tails = np.full(1 + self.k_negative, fill_value=tail, dtype=np.int64)
            if self.k_negative - num_replace_head > 0:
                tails[1 + num_replace_head:] = self.sample_neighbors(
                    [head_tail_neighbors, head_only_neighbors, tail_only_neighbors],
                    self.get_n_samples_per_category(self.k_negative - num_replace_head),
                )
            item['tail'] = tails
            item['target'] = 0
            replace_heads = np.zeros(self.k_negative, dtype=np.int64)
            if self.k_negative - num_replace_head > 0:
                replace_heads[:num_replace_head] = 1
            item['replace_heads'] = replace_heads

        return item

    def __len__(self):
        return len(self.graph)

    def num_tokens(self, index):
        return self.graph.sizes[index]

    def size(self, index):
        return self.graph.sizes[index]

    @property
    def sizes(self):
        return self.graph.sizes

    def ordered_indices(self):
        return self.graph.ordered_indices()