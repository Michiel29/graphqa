import logging
import numpy as np

from fairseq.data import FairseqDataset
from fairseq.data.token_block_utils_fast import _get_slice_indices_fast

from datasets import AnnotatedText
from utils.data_utils import numpy_seed
from utils.plasma_utils import maybe_move_to_plasma


logger = logging.getLogger(__name__)


class TokenBlockAnnotatedDataset(FairseqDataset):
    def __init__(
        self,
        annotated_text,
        max_positions,
        pad,
        eos,
        document_sep_len,
        seed,
    ):
        super().__init__()
        self.dataset = annotated_text
        self.pad = pad
        self.eos = eos
        self.max_positions = max_positions

        assert len(self.dataset) > 0
        sizes = annotated_text.sizes.astype(np.int64)

        slice_indices = _get_slice_indices_fast(
            sizes=sizes,
            break_mode='complete_doc',
            block_size=self.max_positions,
            document_sep_len=document_sep_len,
        )
        _sizes = slice_indices[:, 1] - slice_indices[:, 0]

        self._slice_indices = maybe_move_to_plasma(slice_indices)
        self._sizes = maybe_move_to_plasma(_sizes)

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
            start_block, end_block = self._slice_indices.array[index]
            end_block = min(start_block + self.max_positions, end_block)

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

    def __len__(self):
        return len(self._slice_indices.array)

    @property
    def supports_prefetch(self):
        return False