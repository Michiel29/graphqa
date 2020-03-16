import torch

import numpy as np
import numpy.random as rd

from fairseq.data import FairseqDataset
from fairseq.data import data_utils

from datasets import (
    AnnotatedTextDataset,
    subsample_graph_by_entity_pairs,
)


class RelInfDataset(FairseqDataset):

    def __init__(
        self,
        annotated_text_dataset,
        dictionary,
        graph,
        k_negative,
        n_entities,
        subsampling_strategy,
        subsampling_cap,
        max_positions,
        seed,
    ):
        self.annotated_text_dataset = annotated_text_dataset
        self.k_negative = k_negative
        self.n_entities = n_entities
        self.graph = graph
        self.seed = seed
        self.dataset = annotated_text_dataset
        self.dictionary = dictionary
        self.subsampling_strategy = subsampling_strategy
        self.subsampling_cap = subsampling_cap
        self.max_positions = max_positions
        self.epoch = None

    def set_epoch(self, epoch):
        if self.epoch != epoch:
            self.epoch = epoch
            if self.subsampling_strategy == 'by_entity_pair':
                assert self.subsampling_cap is not None
                with data_utils.numpy_seed(17011990, self.seed, self.epoch):
                    self.dataset = subsample_graph_by_entity_pairs(
                        self.annotated_text_dataset,
                        self.graph,
                        self.subsampling_cap,
                        self.max_positions,
                    )
            else:
                assert self.subsampling_strategy is None

    def __getitem__(self, index):
        item = self.dataset[index]
        head = item['head']
        tail = item['tail']

        with data_utils.numpy_seed(17101990, self.seed, self.epoch, index):
            replace_heads = rd.randint(2, size=self.k_negative)

            head_neighbors = self.graph[head]['neighbors']
            tail_neighbors = self.graph[tail]['neighbors']

            tail_head_neighbors = [tail_neighbors, head_neighbors]

            replacement_entities = []

            for replace_head in replace_heads:
                replacement_neighbors, static_neighbors = tail_head_neighbors[replace_head], tail_head_neighbors[1 - replace_head]

                if len(replacement_neighbors) > 0:
                    replacement_entity = rd.choice(replacement_neighbors)
                elif len(static_neighbors) > 0:
                    replacement_entity = rd.choice(static_neighbors)
                else:
                    replacement_entity = rd.randint(self.n_entities)

                replacement_entities.append(replacement_entity)

            item['head'] = [head] + [head if not replace_heads[i] else replacement_entities[i] for i in range(self.k_negative)]
            item['tail'] = [tail] + [tail if replace_heads[i] else replacement_entities[i] for i in range(self.k_negative)]
            item['target'] = 0

        return item

    def __len__(self):
        return len(self.dataset)

    def num_tokens(self, index):
        return self.dataset.sizes[index]

    def size(self, index):
        return self.dataset.sizes[index]

    @property
    def sizes(self):
        return self.dataset.sizes

    def ordered_indices(self):
        """Sorts by sentence length, randomly shuffled within sentences of same length"""
        return np.lexsort([
            np.random.permutation(len(self)),
            self.dataset.sizes,
        ])