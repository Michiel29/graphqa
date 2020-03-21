import torch

import numpy as np
import numpy.random as rd

class AnnotatedText(object):

    def __init__(
        self,
        text_data,
        annotation_data,
        dictionary,
        entity_dictionary,
        mask_type,
        alpha,
    ):
        self.text_data = text_data
        self.annotation_data = annotation_data
        self.entity_dictionary = entity_dictionary
        self.dictionary = dictionary
        self.mask_type = mask_type
        assert self.mask_type in ['head_tail', 'start_end', None]
        if self.mask_type == 'start_end':
            self.start_tokens = [self.dictionary.e1_start(), self.dictionary.e2_start()]
            self.end_tokens = [self.dictionary.e1_end(), self.dictionary.e2_end()]
        self.alpha = alpha

    def annotate(self, tail_entity, head_entity, head_start_pos, head_end_pos, tail_start_pos, tail_end_pos, start_block, end_block):
        text = np.frombuffer(
            self.text_data._bin_buffer,
            dtype=self.text_data._index.dtype,
            count=end_block - start_block,
            offset=start_block * self.text_data._index.dtype().itemsize,
        )
        text = text.astype(np.int64)

        if self.mask_type is None:
            return torch.tensor(text)

        annotations = self.annotations_block(start_block, end_block)
        # both head and tail should be in annotations
        assert len(annotations) >= 2

        if self.mask_type == 'head_tail':
            text = self.head_tail_mask(
                text,
                annotations,
                head_entity,
                tail_entity,
                head_start_pos,
                head_end_pos,
                tail_start_pos,
                tail_end_pos,
                start_block,
            )
        elif self.mask_type == 'start_end':
            assert False
            text = self.start_end_mask(text, annotations, head_entity, tail_entity)
        else:
            raise Exception('Unknown mask type: %s' % self.mask_type)

        text = torch.tensor(text)
        return text

    def annotations_block(self, start_block, end_block):
        # We are interested in all annotations that INTERSECT [start_block; end_block)
        # Recall that the [start_pos; end_pos) interval for the annotation s is defined as
        # [annotations[s - 1][0], annotations[s - 1][1])
        #
        # From https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html
        # side	returned index i satisfies
        # left	a[i-1] < v <= a[i]
        # right	a[i-1] <= v < a[i]
        #
        # First, we need to find an index s such that
        # annotations[s - 1].end_pos <= start_block < annotations[s].end_pos
        s = np.searchsorted(self.annotation_data[:, 1], start_block, side='right')

        # It's possible that if start_block is so large that s for which
        # start_block < annotations[s].end_pos does not exist.
        # In this case, searchsorted will return len(self.annotation_data)
        # and we need to return an empty list
        if s == len(self.annotation_data):
            return []

        # Second, we need to find an index e such that
        # annotations[e - 1].start_pos < end_block <= annotations[e].start_pos
        e = np.searchsorted(self.annotation_data[:, 0], end_block, side='left')

        # It's possible that if start_block is so small that e for which
        # annotations[e - 1].start_pos < end_block does not exists.
        # In this case, searchsorted will return 0
        # and we need to return an empty list
        assert e < len(self.annotation_data)
        if e == 0:
            return []

        return self.annotation_data[slice(s, e)]

    def head_tail_mask(
        self,
        text,
        annotations,
        head_entity,
        tail_entity,
        head_start_pos,
        head_end_pos,
        tail_start_pos,
        tail_end_pos,
        start_block,
    ):
        num_target_annotations_found = 0
        entity_replacement = {
            head_start_pos: self.dictionary.head(),
            tail_start_pos: self.dictionary.tail(),
        }
        other_entity_replacement = {
            head_entity: self.dictionary.blank_head_other(),
            tail_entity: self.dictionary.blank_tail_other(),
        }

        for annotation in annotations:
            annotation_start_pos = annotation[0]
            annotation_start_pos_local = max(annotation_start_pos - start_block, 0)
            annotation_slice_local = slice(
                annotation_start_pos_local,
                min(annotation[1] - start_block, len(text)),
            )
            annotation_entity = annotation[2]

            if annotation_start_pos in entity_replacement:
                text[annotation_slice_local] = -1
                text[annotation_start_pos_local] = entity_replacement[annotation_start_pos]
                num_target_annotations_found += 1
            elif annotation_entity in other_entity_replacement:
                text[annotation_slice_local] = -1
                text[annotation_start_pos_local] = other_entity_replacement[annotation_entity]

        # both head and tail should be in annotations
        assert num_target_annotations_found == 2
        text = text[text != -1]

        return text

    def start_end_mask(
        self,
        text,
        annotations,
        head_entity,
        tail_entity,
        head_start_pos,
        head_end_pos,
        tail_start_pos,
        tail_end_pos,
        start_block,
    ):
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
