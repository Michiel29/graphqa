import torch

import numpy as np
import numpy.random as rd

from fairseq.data import data_utils, FairseqDataset


class AnnotatedTextDataset(FairseqDataset):

    def __init__(
        self,
        text_data,
        annotation_data,
        dictionary,
        shift_annotations,
        mask_type,
        assign_head_tail,
        seed,
        alpha,
    ):
        self.text_data = text_data
        self.annotation_data = annotation_data
        assert len(self.text_data) == len(self.annotation_data)
        self.shift_annotations = shift_annotations
        self.dictionary = dictionary
        self.mask_type = mask_type
        assert self.mask_type in ['head_tail', 'start_end', None]
        self.assign_head_tail = assign_head_tail
        assert self.assign_head_tail in ['random', 'first', None]
        self.alpha = alpha
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index, head_entity=None, tail_entity=None):
        text = self.text_data[index]
        annotations = self.annotation_data[index]

        with data_utils.numpy_seed(hash(self.__class__), self.seed, self.epoch, index):
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

    def start_end_mask(self, text, annotations, e1_temp, e2_temp):
        annotations_list = annotations.split(3)
        entity_ids = annotations[2::3].numpy()
        unique_entity_ids = np.unique(entity_ids)
        assert len(unique_entity_ids) >= 2

        # Get e1 and e2 indices (directed)
        e1, e2 = e1_temp, e2_temp
        e1_indices = np.where(entity_ids == e1)[0]
        e2_indices = np.where(entity_ids == e2)[0]
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

        # Get e1 and e2 start/end indices
        e1_annotation = annotations_list[e1_idx][2].item()
        e1_start = annotations_list[e1_idx][0].item() + self.shift_annotations
        e1_end = annotations_list[e1_idx][1].item() + self.shift_annotations

        e2_annotation = annotations_list[e2_idx][2].item()
        e2_start = annotations_list[e2_idx][0].item() + self.shift_annotations
        e2_end = annotations_list[e2_idx][1].item() + self.shift_annotations

        # Initialize new text with -1's
        text_new = -1 * torch.ones(text.shape[0]+4).long()

        # Copy over non-entity tokens from original text to new text
        text_new[:e1_start] = text[:e1_start]
        text_new[e1_end+2:e2_start+2] = text[e1_end:e2_start]
        text_new[e2_end+4:] = text[e2_end:]

        # Insert e1 and e2 start/end tokens into new text
        text_new[e1_start] = self.dictionary.e1_start()
        text_new[e1_end+1] = self.dictionary.e1_end()
        text_new[e2_start+2] = self.dictionary.e2_start()
        text_new[e2_end+3] = self.dictionary.e2_end()

        # For each entity, randomly decide whether to mask it with a [BLANK] token
        #   - NO, with probability alpha
        #   - YES, with probability 1-alpha
        mask_decision = np.random.choice(2, 2, p=[self.alpha, 1 - self.alpha])

        if mask_decision[0] == 1:
            text_new[e1_start+1] = self.dictionary.blank()
        else:
            text_new[e1_start+1:e1_end+1] = text[e1_start:e1_end]

        if mask_decision[1] == 1:
            text_new[e2_start+3] = self.dictionary.blank()
        else:
            text_new[e2_start+3:e2_end+3] = text[e2_start:e2_end]

        # Remove any -1's in new text left over after [BLANK] masking
        text_new = text_new[text_new != -1]

        return {
            'text': text_new,
            'head': e1,
            'tail': e2,
            'ntokens': len(text_new),
            'nsentences': 1,
        }
