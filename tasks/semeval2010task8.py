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
    AnnotatedText,
    SemEval2010Task8Dataset,
    FilteredDataset,
    PrependTokenDataset,
    prune_dataset_size,
)
from utils.data_utils import (
    load_annotated_text,
    safe_load_indexed_dataset,
)
from utils.dictionary import CustomDictionary

logger = logging.getLogger(__name__)


@register_task('semeval2010task8')
class SemEval2010Task8Task(BaseTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data_path', help='path to data')

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = CustomDictionary.load(dict_path)
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary, None)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        text_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.text'),
        )
        annotation_data = np.load(
            os.path.join(self.args.data_path, split + '.annotations.npy'),
            mmap_mode='r',
        )
        annotated_text = AnnotatedText(
            text_data=text_data,
            annotation_data=annotation_data,
            dictionary=self.dictionary,
            mask_type=self.args.mask_type,
            non_mask_rate=self.args.non_mask_rate,
        )
        relation_dataset = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.relations')
        )

        dataset = SemEval2010Task8Dataset(
            annotation_text=annotated_text,
            relation_dataset=relation_dataset,
            dictionary=self.dictionary,
            seed=self.seed,
        )
        dataset = PrependTokenDataset(dataset, self.dictionary.bos(), ['text'])

        n_examples = getattr(self.args, 'n_' + split + '_examples', None)
        if n_examples is not None:
            dataset = prune_dataset_size(
                dataset,
                n_examples,
                self.seed,
            )

        self.datasets[split] = dataset
