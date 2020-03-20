import logging
import os

import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    FairseqDataset,
    iterators,
)
from fairseq.tasks import register_task

from tasks import BaseTask
from datasets import (
    AnnotatedTextDataset,
    FewRelDataset,
    FilteredDataset,
    prune_dataset_size,
)
from utils.data_utils import (
    load_annotated_text,
    safe_load_indexed_dataset,
)
from utils.dictionary import CustomDictionary

logger = logging.getLogger(__name__)


@register_task('fewrel')
class FewRelTask(BaseTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data_path', help='path to data')
        parser.add_argument('--n_way', default=5, help='number of few-shot classes')
        parser.add_argument('--n_shot', default=1, help='number of few-shot examples')

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = CustomDictionary.load(dict_path)
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary, None)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        text_data, annotation_data = load_annotated_text(
            self.args.data_path,
            split,
            self.dictionary.bos(),
        )
        relation_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.relations')
        )
        annotated_text_dataset = AnnotatedTextDataset(
            text_data=text_data,
            annotation_data=annotation_data,
            dictionary=self.dictionary,
            entity_dictionary=self.entity_dictionary,
            shift_annotations=1,
            mask_type=self.args.mask_type,
            assign_head_tail='first',
            seed=self.seed,
            alpha=self.args.alpha,
        )
        annotated_text_dataset, indices = self.filter_by_max_positions(
            annotated_text_dataset,
            return_indices=True,
        )
        relation_dataset = FilteredDataset(relation_data, indices)

        n_examples = int(getattr(self.args, 'n_' + split + '_examples'))
        if n_examples > 0:
            annotated_text_dataset, indices = prune_dataset_size(
                annotated_text_dataset,
                n_examples,
                self.seed,
                return_indices=True,
            )
            relation_dataset = FilteredDataset(relation_dataset, indices)

        dataset = FewRelDataset(
            annotation_text_dataset=annotated_text_dataset,
            relation_dataset=relation_dataset,
            dictionary=self.dictionary,
            entity_dictionary=self.entity_dictionary,
            mask_type=self.mask_type,
            n_way=self.args.n_way,
            n_shot=self.args.n_shot,
            seed=self.seed,
        )

        self.datasets[split] = dataset
