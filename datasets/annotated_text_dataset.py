import torch

import numpy as np
import numpy.random as rd

from fairseq.data import data_utils, FairseqDataset
from utils.diagnostic_utils import Diagnostic

class AnnotatedTextDataset(FairseqDataset):

    def __init__(
        self,
        text_data,
        annotation_data,
        dictionary,
        entity_dictionary,
        shift_annotations,
        mask_type,
        assign_head_tail,
        seed,
        alpha=None,
    ):
        self.text_data = text_data
        self.annotation_data = annotation_data
        assert len(self.text_data) == len(self.annotation_data)
        self.shift_annotations = shift_annotations
        self.dictionary = dictionary
        self.mask_type = mask_type
        assert self.mask_type in ['head_tail', 'start_end', None]
        if self.mask_type == 'start_end':
            self.start_tokens = [self.dictionary.e1_start(), self.dictionary.e2_start()]
            self.end_tokens = [self.dictionary.e1_end(), self.dictionary.e2_end()]
        self.assign_head_tail = assign_head_tail
        assert self.assign_head_tail in ['random', 'first', None]
        self.alpha = alpha
        self.seed = seed
        self.epoch = 0

        #self.diag = Diagnostic(dictionary, entity_dictionary)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index, head_entity=None, tail_entity=None):
        text = self.text_data[index]
        annotations = self.annotation_data[index]

        with data_utils.numpy_seed(9031934, self.seed, self.epoch, index):
            if head_entity is None:
                assert tail_entity is None
                if self.assign_head_tail == 'random':
                    head_entity, tail_entity = self.assign_head_tail_randomly(annotations)
                elif self.assign_head_tail == 'first':
                    head_entity, tail_entity = self.assign_head_tail_first(annotations)
            else:
                assert tail_entity is not None

            if self.mask_type == 'head_tail':
                item = self.head_tail_mask(text, annotations, head_entity, tail_entity)
            elif self.mask_type == 'start_end':
                item = self.start_end_mask(text, annotations, head_entity, tail_entity)
            else:
                item = {
                    'text': text,
                    'ntokens': len(text),
                    'nsentences': 1,
                }

        # self.diag.inspect_item(item['text'], head_entity, tail_entity)

        return item

    def __len__(self):
        return len(self.text_data)

    def num_tokens(self, index):
        return self.text_data.sizes[index]

    def size(self, index):
        return self.text_data.sizes[index]

    @property
    def sizes(self):
        return self.text_data.sizes

    def ordered_indices(self):
        """Sorts by sentence length, randomly shuffled within sentences of same length"""
        return np.lexsort([
            np.random.permutation(len(self)),
            self.text_data.sizes,
        ])

    def assign_head_tail_randomly(self, annotations):
        annotations = annotations.split(3)
        unique_entity_ids = np.unique([annotation[2] for annotation in annotations])
        assert len(unique_entity_ids) >= 2
        head_entity, tail_entity = np.random.choice(
            unique_entity_ids,
            size=2,
            replace=False,
        )
        return head_entity, tail_entity

    def assign_head_tail_first(self, annotations):
        annotations = annotations.split(3)
        unique_entity_ids = np.unique([annotation[2] for annotation in annotations])
        assert len(unique_entity_ids) >= 2
        return unique_entity_ids[:2]

    def head_tail_mask(self, text, annotations, head_entity, tail_entity):
        entity_replacement = {
            head_entity: self.dictionary.head(),
            tail_entity: self.dictionary.tail(),
        }

        annotations = annotations.split(3)
        for annotation in annotations:
            annotation_entity = annotation[2].item()
            if annotation_entity in entity_replacement:
                ent_start = annotation[0].item() + self.shift_annotations
                ent_end = annotation[1].item() + self.shift_annotations
                text[ent_start:ent_end] = -1
                text[ent_start] = entity_replacement[annotation_entity]

        text = text[text != -1]

        return {
            'text': text,
            'head': head_entity,
            'tail': tail_entity,
            'ntokens': len(text),
            'nsentences': 1,
        }

    def start_end_mask(self, text, annotations, head_entity, tail_entity):
        annotations_list = annotations.split(3)
        entity_ids = annotations[2::3].numpy()

        # Get e1 and e2 indices (directed)
        e1_indices = np.where(entity_ids == head_entity)[0]
        e2_indices = np.where(entity_ids == tail_entity)[0]
        e1_idx = np.random.choice(e1_indices)
        e2_idx = np.random.choice(e2_indices)

        # Get e1 and e2 indices (undirected) -- MTB paper does this, but it invalidates our directed case1 pre-filtering
        '''
        e1_indices = np.where(entity_ids == e1_temp)[0]
        e2_indices = np.where(entity_ids == e2_temp)[0]
        e1_temp_idx = np.random.choice(e1_indices)
        e2_temp_idx = np.random.choice(e2_indices)
        if e1_temp_idx < e2_temp_idx:
            e1 = e1_temp
            e1_idx = e1_temp_idx
            e2 = e2_temp
            e2_idx = e2_temp_idx
        else:
            e1 = e2_temp
            e1_idx = e2_temp_idx
            e2 = e1_temp
            e2_idx = e1_temp_idx
        '''
        
        # For each entity, randomly decide whether to mask it with a [BLANK] token
        #   - NO, with probability alpha
        #   - YES, with probability 1-alpha
        mask_decision = np.random.choice(2, 2, p=[self.alpha, 1 - self.alpha])

        # Get e1 and e2 start/end indices - we assume intervals don't overlap
        start_entity_events = {
            annotations_list[e1_idx][0].item() + self.shift_annotations: 0,
            annotations_list[e2_idx][0].item() + self.shift_annotations: 1,
        }
        end_entity_events = {
            annotations_list[e1_idx][1].item() + self.shift_annotations - 1: 0,
            annotations_list[e2_idx][1].item() + self.shift_annotations - 1: 1,
        }
        current_entity = None
        text_new = []
        for text_index in range(len(text)):
            if text_index in start_entity_events:
                current_entity = start_entity_events[text_index]
                text_new.append(self.start_tokens[current_entity])
            # Check if we need to replace entity surface form with [BLANK]
            if current_entity is not None and mask_decision[current_entity]:
                # Check if we have already inserted [BLANK]
                assert len(text_new) > 0
                if text_new[-1] != self.dictionary.blank():
                    text_new.append(self.dictionary.blank())
            else:
                text_new.append(text[text_index])
            if text_index in end_entity_events:
                assert current_entity == end_entity_events[text_index]
                text_new.append(self.end_tokens[current_entity])
                current_entity = None

        if mask_decision.sum() == 0:
            assert len(text_new) == len(text) + 4

        text_new = torch.LongTensor(text_new)
        assert (text_new == self.dictionary.blank()).sum() == mask_decision.sum()

        assert self.dictionary.e1_start() in text_new
        assert self.dictionary.e1_end() in text_new
        assert self.dictionary.e2_start() in text_new
        assert self.dictionary.e2_end() in text_new

        return {
            'text': text_new,
            'head': head_entity,
            'tail': tail_entity,
            'ntokens': len(text_new),
            'nsentences': 1,
        }
