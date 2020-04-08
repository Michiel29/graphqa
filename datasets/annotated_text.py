import logging
import numpy as np
import time
import torch

from utils.plasma_utils import maybe_move_to_plasma


logger = logging.getLogger(__name__)


class AnnotatedText(object):

    INDEX_ANNOTATION_START = 0
    INDEX_ANNOTATION_END = 1
    INDEX_ANNOTATION_ENTITY = 4

    def __init__(
        self,
        text_data,
        annotation_data,
        dictionary,
        mask_type,
        non_mask_rate,
    ):
        start_time = time.time()
        self.text_data = text_data
        self.annotation_data = annotation_data

        self.dictionary = dictionary
        self.mask_type = mask_type
        assert self.mask_type in ['head_tail', 'start_end', None]
        self.non_mask_rate = non_mask_rate

        offsets = np.roll(np.cumsum(self.text_data._index._sizes), 1)
        offsets[0] = 0
        self.sentence_offsets = maybe_move_to_plasma(offsets)
        logger.info('set up annotated text [n_sentences=%d, n_annotations=%d, mask_type=%s] in %.3f seconds' % (
            len(self.text_data),
            len(self.annotation_data.array),
            self.mask_type,
            time.time() - start_time,
        ))

    def annotate_sentence(self, index, head_entity, tail_entity):
        start_block = self.sentence_offsets.array[index]
        end_block = start_block + self.text_data._index._sizes[index]
        annotations = self.annotations_block(start_block, end_block)
        head_annotation = self.sample_annotation(annotations, head_entity)
        tail_annotation = self.sample_annotation(annotations, tail_entity)
        return self.annotate(
            tail_entity=tail_entity,
            head_entity=head_entity,
            head_start_pos=head_annotation[self.INDEX_ANNOTATION_START],
            head_end_pos=head_annotation[self.INDEX_ANNOTATION_END],
            tail_start_pos=tail_annotation[self.INDEX_ANNOTATION_START],
            tail_end_pos=tail_annotation[self.INDEX_ANNOTATION_END],
            start_block=start_block,
            end_block=end_block,
        )

    def annotate(
        self,
        tail_entity,
        head_entity,
        head_start_pos,
        head_end_pos,
        tail_start_pos,
        tail_end_pos,
        start_block,
        end_block,
        annotations=None,
    ):
        text = np.frombuffer(
            self.text_data._bin_buffer,
            dtype=self.text_data._index.dtype,
            count=end_block - start_block,
            offset=start_block * self.text_data._index.dtype().itemsize,
        )
        text = text.astype(np.int64)

        if self.mask_type is None:
            return torch.tensor(text)

        if annotations is None:
            annotations = self.annotations_block(start_block, end_block)

        annotations = self.filter_by_entities(annotations, {head_entity, tail_entity})
        # both head and tail should be in annotations
        assert len(annotations) >= int(head_entity is not None) + int(tail_entity is not None)
        if len(annotations) > 0:
            annotations[:, self.INDEX_ANNOTATION_START] = np.maximum(
                annotations[:, self.INDEX_ANNOTATION_START] - start_block,
                0,
            )
            annotations[:, self.INDEX_ANNOTATION_END] = np.minimum(
                annotations[:, self.INDEX_ANNOTATION_END] - start_block,
                len(text),
            )
        if head_entity is not None:
            head_start_pos -= start_block
            head_end_pos -= start_block
        if tail_entity is not None:
            tail_start_pos -= start_block
            tail_end_pos -= start_block

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
            )
        elif self.mask_type == 'start_end':
            text = self.start_end_mask(
                text,
                annotations,
                head_entity,
                tail_entity,
                head_start_pos,
                head_end_pos,
                tail_start_pos,
                tail_end_pos,
            )
        else:
            raise Exception('Unknown mask type: %s' % self.mask_type)

        text = torch.tensor(text)
        return text

    def annotations_block(self, start_block, end_block):
        # From http://sociograph.blogspot.com/2011/12/gotcha-with-numpys-searchsorted.html
        start_block = self.annotation_data.array.dtype.type(start_block)
        end_block = self.annotation_data.array.dtype.type(end_block)

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
        s = np.searchsorted(self.annotation_data.array[:, 1], start_block, side='right')

        # It's possible that if start_block is so large that s for which
        # start_block < annotations[s].end_pos does not exist.
        # In this case, searchsorted will return len(self.annotation_data)
        # and we need to return an empty list
        if s == len(self.annotation_data.array):
            return []

        # Second, we need to find an index e such that
        # annotations[e - 1].start_pos < end_block <= annotations[e].start_pos
        e = np.searchsorted(self.annotation_data.array[:, 0], end_block, side='left')

        # It's possible that if start_block is so small that e for which
        # annotations[e - 1].start_pos < end_block does not exists.
        # In this case, searchsorted will return 0
        # and we need to return an empty list
        if e == 0:
            return []

        return self.annotation_data.array[slice(s, e)]

    def filter_by_entities(self, annotations, entity_set):
        annotations_new = list()
        for annotation in annotations:
            if annotation[self.INDEX_ANNOTATION_ENTITY] in entity_set:
                annotations_new.append(annotation)
        if len(annotations_new) > 0:
            return np.stack(annotations_new)
        else:
            return np.array([], dtype=np.int64)

    def sample_annotation(self, annotations, entity):
        annotations = self.filter_by_entities(annotations, {entity})
        assert len(annotations) > 0
        index = np.random.randint(len(annotations))
        return annotations[index]

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
    ):
        text = self.mask_annotations(
            text,
            annotations,
            head_entity,
            self.dictionary.blank_head_other(),
            target_annotation_start=head_start_pos,
        )
        text = self.mask_annotations(
            text,
            annotations,
            tail_entity,
            self.dictionary.blank_tail_other(),
            target_annotation_start=tail_start_pos,
        )
        text[slice(head_start_pos, head_end_pos)] = -1
        text[head_start_pos] = self.dictionary.head()
        text[slice(tail_start_pos, tail_end_pos)] = -1
        text[tail_start_pos] = self.dictionary.tail()

        text = text[text != -1]

        assert (
            (text == self.dictionary.blank_head_other()).sum()
            + (text == self.dictionary.blank_tail_other()).sum()
            == len(annotations) - 2
        )
        return text

    def prepare_replacements(
        self,
        text,
        annotations,
        entity_id,
        target_annotation_start,
        marker_start,
        marker_end,
        blank_other,
    ):
        if entity_id is None:
            return []
        # For each entity, randomly decide whether to mask it with a [BLANK] token
        #   - NO, with probability non_mask_rate
        #   - YES, with probability 1-non_mask_rate
        mask_decision = np.random.choice(2, 1, p=[self.non_mask_rate, 1 - self.non_mask_rate])[0]
        replacements = []

        for annotation in annotations:
            annotation_start_pos = annotation[self.INDEX_ANNOTATION_START]
            annotation_end_pos = annotation[self.INDEX_ANNOTATION_END]
            if annotation[self.INDEX_ANNOTATION_ENTITY] == entity_id:
                is_target_annotation = (annotation_start_pos == target_annotation_start)
                if is_target_annotation:
                    if mask_decision:
                        replacements.append((
                            annotation_start_pos,
                            annotation_end_pos,
                            [marker_start, self.dictionary.blank(), marker_end],
                        ))
                    else:
                        replacements.append((
                            annotation_start_pos,
                            annotation_end_pos,
                            np.concatenate(([marker_start], text[annotation_start_pos:annotation_end_pos], [marker_end]))
                        ))
                else:
                    if mask_decision:
                        replacements.append((
                            annotation_start_pos,
                            annotation_end_pos,
                            [blank_other],
                        ))
        assert len(replacements) >= 1
        return replacements

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
    ):
        replacements = self.prepare_replacements(
            text,
            annotations,
            head_entity,
            head_start_pos,
            self.dictionary.e1_start(),
            self.dictionary.e1_end(),
            self.dictionary.blank_head_other(),
        ) + self.prepare_replacements(
            text,
            annotations,
            tail_entity,
            tail_start_pos,
            self.dictionary.e2_start(),
            self.dictionary.e2_end(),
            self.dictionary.blank_tail_other(),
        )
        replacements.sort()
        replacements = reversed(replacements)

        for start_pos, end_pos, replacement_tokens in replacements:
            text = np.concatenate([
                text[:start_pos],
                replacement_tokens,
                text[end_pos:],
            ])

        if head_entity is not None:
            assert (text == self.dictionary.e1_start()).sum() == 1
            assert (text == self.dictionary.e1_end()).sum() == 1
        if tail_entity is not None:
            assert (text == self.dictionary.e2_start()).sum() == 1
            assert (text == self.dictionary.e2_end()).sum() == 1

        return text

    def __len__(self):
        return len(self.text_data)

    def num_tokens(self, index):
        return self.text_data.sizes[index]

    def size(self, index):
        return self.text_data.sizes[index]

    @property
    def sizes(self):
        return self.text_data.sizes
