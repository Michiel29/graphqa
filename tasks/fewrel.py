import logging
import os

import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    iterators,
    FairseqDataset,
    PrependTokenDataset
)
from fairseq.tasks import register_task

from tasks import BaseTask
from datasets import FewRelDataset, FixedSizeDataset

from utils.data_utils import CustomDictionary

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
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        assert split in ['train', 'test', 'valid']
        split_path = os.path.join(self.args.data_path, split)

        text_path = os.path.join(self.args.data_path, split + '.text')
        annotation_path = os.path.join(self.args.data_path, split + '.annotations')
        relation_path = os.path.join(self.args.data_path, split + '.relations')

        text_data =  data_utils.load_indexed_dataset(
            text_path,
            None,
            dataset_impl='mmap',
        )

        if text_data is None:
            raise FileNotFoundError('Dataset (text) not found: {}'.format(text_path))

        text_data = PrependTokenDataset(text_data, self.dictionary.bos())

        annotation_data = data_utils.load_indexed_dataset(
            annotation_path,
            None,
            dataset_impl='mmap',
        )

        if annotation_data is None:
            raise FileNotFoundError('Dataset (annotation) not found: {}'.format(annotation_path))

        relation_data = data_utils.load_indexed_dataset(
            relation_path,
            None,
            dataset_impl='mmap',
        )

        if relation_data is None:
            raise FileNotFoundError('Dataset (relations) not found: {}'.format(relation_path))

        n_examples = int(getattr(self.args, 'n_' + split + '_examples', -1))

        text_data = FixedSizeDataset(text_data, n_examples)
        annotation_data = FixedSizeDataset(annotation_data, n_examples)
        relation_data = FixedSizeDataset(relation_data, n_examples)

        dataset = FewRelDataset(
            text_data=text_data,
            annotation_data=annotation_data,
            relation_data=relation_data,
            dictionary=self.dictionary,
            mask_type=self.mask_type,
            n_way=self.args.n_way,
            n_shot=self.args.n_shot,
            # TODO(urikz): Remove this
            dataset_size=n_examples,
            shift_annotations=1, # because of the PrependTokenDataset
            seed=self.seed,
        )

        self.datasets[split] = dataset
