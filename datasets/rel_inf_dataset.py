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
    ):
        super().__init__(
            text_data,
            annotation_data,
            dictionary,
            shift_annotations,
            assign_head_tail_randomly=True,
        )
        self.k_negative = k_negative
        self.n_entities = n_entities
        self.graph = graph

    def __getitem__(self, index):
        item = super().__getitem__(index)
        head = item['head']
        tail = item['tail']

        replace_head_entity = rd.randint(2, size=self.k_negative)

        tail_head = np.array([tail, head])
        entities_to_be_replaced = tail_head[replace_head_entity]
        entities_not_replaced = tail_head[1 - replace_head_entity]
        replacement_entities = []
        for entity_replaced, entity_not_replaced in zip(entities_to_be_replaced, entities_not_replaced):
            if len(self.graph.entity_neighbors[entity_not_replaced]) > 0:
                replacement_entity = rd.choice(list(self.graph.entity_neighbors[entity_not_replaced].keys()))
            elif len(self.graph.entity_neighbors[entity_replaced]) > 0:
                replacement_entity = rd.choice(list(self.graph.entity_neighbors[entity_replaced].keys()))
            else:
                replacement_entity = rd.randint(self.n_entities)

            replacement_entities.append(replacement_entity)

        item['head'] = [head] + [head if replace_head_entity[i] else replacement_entities[i] for i in range(self.k_negative)]
        item['tail'] = [tail] + [tail if not replace_head_entity[i] else replacement_entities[i] for i in range(self.k_negative)]
        item['target'] = 0

        return item

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
