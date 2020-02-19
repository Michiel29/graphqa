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

    def __getitem__(self, index):
        item = super().__getitem__(index)
        head = item['head']
        tail = item['tail']

        replace_head_entity = rd.randint(2, size=self.k_negative)
        replacement_entities = rd.randint(self.n_entities, size=self.k_negative)

        item['head'] = [head] + [head if replace_head_entity[i] else replacement_entities[i] for i in range(self.k_negative)]
        item['tail'] = [tail] + [tail if not replace_head_entity[i] else replacement_entities[i] for i in range(self.k_negative)]
        item['target'] = [0] * (self.k_negative + 1)

        return item

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
