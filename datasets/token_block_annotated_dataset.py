import logging
import numpy as np

from fairseq.data import TokenBlockDataset

from datasets import AnnotatedText
from utils.data_utils import numpy_seed


logger = logging.getLogger(__name__)


class TokenBlockAnnotatedDataset(TokenBlockDataset):
    def __init__(
        self,
        annotated_text,
        max_positions,
        pad,
        eos,
        document_sep_len,
        seed,
    ):
        super().__init__(
            dataset=annotated_text,
            sizes=annotated_text.sizes,
            block_size=max_positions,
            pad=pad,
            eos=eos,
            break_mode='complete_doc',
            include_targets=False,
            document_sep_len=document_sep_len,
        )
        self.max_positions = max_positions
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample_entities(self, entities):
        if len(entities) == 0:
            return None, None
        if len(entities) == 1:
            if np.random.randint(2):
                return entities[0], None
            else:
                return None, entities[0]
        return np.random.choice(entities, 2, replace=False)

    def sample_annotation(self, annotations, entity):
        if entity is None:
            return None, None
        entity_annotation = self.dataset.sample_annotation(annotations, entity)
        return entity_annotation[AnnotatedText.INDEX_ANNOTATION_START], entity_annotation[AnnotatedText.INDEX_ANNOTATION_END]

    def __getitem__(self, index):
        with numpy_seed('TokenBlockAnnotatedDataset', self.seed, self.epoch, index):
            start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
            slice_s, slice_e = self.slice_indices[index]
            length = min(slice_e - slice_s, self.max_positions)

            start_block = self.dataset.sentence_offsets.array[start_ds_idx] + start_offset
            end_block = start_block + length

            annotations = self.dataset.annotations_block(start_block, end_block)
            if len(annotations) > 0:
                entities = np.unique(annotations[:, AnnotatedText.INDEX_ANNOTATION_ENTITY])
            else:
                entities = []
            head_entity, tail_entity = self.sample_entities(entities)
            head_start_pos, head_end_pos = self.sample_annotation(annotations, head_entity)
            tail_start_pos, tail_end_pos = self.sample_annotation(annotations, tail_entity)

            text = self.dataset.annotate(
                tail_entity=tail_entity,
                head_entity=head_entity,
                head_start_pos=head_start_pos,
                head_end_pos=head_end_pos,
                tail_start_pos=tail_start_pos,
                tail_end_pos=tail_end_pos,
                start_block=start_block,
                end_block=end_block,
                annotations=annotations,
            )
            return text

    @property
    def sizes(self):
        # TODO: Note, this is an overestimation of the actual number of tokens
        return np.minimum(self._sizes.array, self.max_positions + 4)

    def num_tokens(self, index):
        # TODO: Note, this is an overestimation of the actual number of tokens
        return min(self._sizes.array[index], self.max_positions + 4)

    def size(self, index):
        # TODO: Note, this is an overestimation of the actual number of tokens
        return min(self._sizes.array[index], self.max_positions + 4)