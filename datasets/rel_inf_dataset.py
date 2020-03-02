import torch

import numpy as np
import numpy.random as rd

from fairseq.data import FairseqDataset

from datasets import AnnotatedTextDataset

class RelInfDataset(AnnotatedTextDataset):

    def __init__(
        self,
        text_data,
        annotation_data,
        graph,
        k_negative,
        n_entities,
        dictionary,
        shift_annotations,
        mask_type,
    ):
        super().__init__(
            text_data=text_data,
            annotation_data=annotation_data,
            dictionary=dictionary,
            shift_annotations=shift_annotations,
            mask_type=mask_type,
            assign_head_tail_randomly=True,
        )
        self.k_negative = k_negative
        self.n_entities = n_entities
        self.graph = graph

    def __getitem__(self, index):
        item = super().__getitem__(index)
        head = item['head']
        tail = item['tail']

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

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
