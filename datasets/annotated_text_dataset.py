import torch

import numpy as np
import numpy.random as rd

from fairseq.data import FairseqDataset


class AnnotatedTextDataset(FairseqDataset):

    def __init__(
        self,
        text_data,
        annotation_data,
        dictionary,
        shift_annotations,
        assign_head_tail_randomly,
    ):
        self.text_data = text_data
        self.annotation_data = annotation_data
        assert len(self.text_data) == len(self.annotation_data)
        self.shift_annotations = shift_annotations
        self.dictionary = dictionary
        self.assign_head_tail_randomly = assign_head_tail_randomly

    def __getitem__(self, index):
        mention = self.text_data[index]
        annotations = self.annotation_data[index].split(3)

        unique_entity_ids = np.unique([annotation[2] for annotation in annotations])

        # TODO(urikz): Fix dataset first
        # assert len(unique_entity_ids) >= 2
        assert len(unique_entity_ids) >= 1

        if self.assign_head_tail_randomly:
            # TODO(urikz): Use replace=False
            head_entity, tail_entity = np.random.choice(unique_entity_ids, size=2)
        else:
            # TODO(urikz): Fix dataset first
            if len(unique_entity_ids) < 2:
                head_entity, tail_entity = unique_entity_ids[0], unique_entity_ids[0]
            else:
                head_entity, tail_entity = unique_entity_ids[:2]

        entity_replacement = {
            head_entity: self.dictionary.head(),
            tail_entity: self.dictionary.tail(),
        }

        for annotation in annotations:
            if annotation[2] in entity_replacement:
                ent_start = annotation[0] + self.shift_annotations
                end_end = annotation[1] + self.shift_annotations
                mention[ent_start:end_end] = -1
                mention[end_start] = entity_replacement[annotation[2]]

        mention = mention[mention != -1]

        return {
            'mention': mention,
            'head': head_entity,
            'tail': tail_entity,
            'ntokens': len(mention),
            'nsentences': 1,
        }

    def __len__(self):
        return len(self.text_data)

    def num_tokens(self, index):
        return self.text_data.sizes[index]

    def size(self, index):
        return self.text_data.sizes[index]

    def ordered_indices(self):
        """Sorts by sentence length, randomly shuffled within sentences of same length"""
        return np.lexsort([
            np.random.permutation(len(self)),
            self.text_data.sizes,
        ])

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
