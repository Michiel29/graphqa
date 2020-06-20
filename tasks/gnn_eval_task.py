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
    GNNEvalDataset,
)
from utils.data_utils import numpy_seed, safe_load_indexed_dataset
from utils.numpy_utils import MMapNumpyArray
from utils.dictionary import CustomDictionary, EntityDictionary

from tasks import BaseTask

logger = logging.getLogger(__name__)

@register_task('gnn_eval')
class GNNEvalTask(BaseTask):

    NUM_SAMPLES_TO_COMPUTE_SAMPLE_SIZE = 20

    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args, dictionary, entity_dictionary)
        self.sample_sizes_mean = None

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data-path', help='path to data')
        parser.add_argument('--mask-type', default='head_tail', type=str,
                            help='method for masking entities in a sentence')
        parser.add_argument('--non-mask-rate', default=1.0, type=float,
                            help='probability of not masking the entity with a [BLANK] token')
        parser.add_argument('--num-text-chunks', type=int, default=None)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = CustomDictionary.load(dict_path)

        entity_dict_path = os.path.join(args.data_path, 'entity.dict.valid.txt')
        if os.path.exists(entity_dict_path):
            entity_dictionary = EntityDictionary.load(entity_dict_path)
            logger.info('entity dictionary: {} types'.format(len(entity_dictionary)))
        else:
            entity_dictionary = None
        logger.info('dictionary: {} types'.format(len(dictionary)))

        task = cls(args, dictionary, entity_dictionary)
        return task

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

        graph_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.graph'),
        )
        graph = GraphDataset(
            edges=graph_data,
            subsampling_strategy=self.args.subsampling_strategy,
            subsampling_cap=self.args.subsampling_cap,
            seed=self.args.seed,
        )

        dataset = GNNEvalDataset(
            annotated_text=annotated_text,
            graph=graph,
            dictionary=self.dictionary,
            max_positions=self.args.max_positions,
            num_workers=self.args.num_workers,
            seed=self.args.seed,
        )

        dataset = PrependTokenDataset(dataset, self.dictionary.bos(), keys=['target', 'support'])

        n_examples = getattr(self.args, 'n_' + split + '_examples', None)
        if n_examples is not None:
            dataset = FixedSizeDataset(
                dataset=dataset,
                size=n_examples,
                seed=self.seed,
            )

        self.datasets[split] = dataset
