from fairseq.data import FairseqDataset, iterators
from fairseq.tasks import register_task

import logging
import numpy as np
import os
import time

from datasets import (
    AnnotatedText,
    EpochSplitDataset,
    FixedSizeDataset,
    GraphDataset,
    PrependTokenDataset,
    ETPRelationDataset,
)
from utils.data_utils import numpy_seed, safe_load_indexed_dataset
from utils.numpy_utils import MMapNumpyArray
from tasks import BaseTask

logger = logging.getLogger(__name__)


@register_task('etp_relation')
class ETPRelation(BaseTask):

    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args, dictionary, entity_dictionary)
        self.sample_sizes_mean = None

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data-path', help='path to data')
        parser.add_argument('--non-mask-rate', default=1.0, type=float,
                            help='probability of not masking the entity with a [BLANK] token')

        parser.add_argument('--total-negatives', type=int, default=None)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        text_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.text'),
        )
        annotation_data = MMapNumpyArray(
            os.path.join(self.args.data_path, split + '.annotations.npy')
        )
        annotated_text = AnnotatedText(
            text_data=text_data,
            annotation_data=annotation_data,
            dictionary=self.dictionary,
            mask_type=self.args.mask_type,
            non_mask_rate=self.args.non_mask_rate,
        )

        edges = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.graph'),
        )

        dataset = ETPRelationDataset(
            annotated_text=annotated_text,
            edges=edges,
            dictionary=self.dictionary,
            n_entities = len(self.entity_dictionary),
            total_negatives=self.args.total_negatives,
            max_positions = self.args.max_positions,
            num_workers=self.args.num_workers,
            seed=self.args.seed,
        )

        if split == 'train' and self.args.epoch_size is not None:
            dataset = EpochSplitDataset(
                dataset=dataset,
                epoch_size=self.args.epoch_size,
                seed=self.args.seed,
            )

        dataset = PrependTokenDataset(dataset, self.dictionary.bos(), 'text', ['annotation'])

        n_examples = getattr(self.args, 'n_' + split + '_examples', None)
        if n_examples is not None:
            dataset = FixedSizeDataset(
                dataset=dataset,
                size=n_examples,
                seed=self.seed,
            )

        self.datasets[split] = dataset
