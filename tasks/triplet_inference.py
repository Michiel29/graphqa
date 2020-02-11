import logging
import os
from collections import namedtuple
import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    iterators,
    FairseqDataset,
    PrependTokenDataset
)
from fairseq.tasks import FairseqTask, register_task

from tasks.relation_inference import RelationInferenceTask
from datasets.triplet_dataset import TripletDataset

logger = logging.getLogger(__name__)

@register_task('triplet_inference')
class TripletInferenceTask(RelationInferenceTask):
    """Task for training inference models."""

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        text_path = os.path.join(self.args.data_path, split + '.text')
        annotation_path = os.path.join(self.args.data_path, split + '.annotations')

        text_data =  data_utils.load_indexed_dataset(
            text_path,
            None,
            dataset_impl='mmap',
        )

        if text_data is None:
            raise FileNotFoundError('Dataset (text) not found: {}'.format(text_path))

        annotation_data =  data_utils.load_indexed_dataset(
            annotation_path,
            None,
            dataset_impl='mmap',
        )        

        if annotation_data is None:
            raise FileNotFoundError('Dataset (annotation) not found: {}'.format(annotation_path))

        text_data = PrependTokenDataset(text_data, self.dictionary.bos())

        if split is 'train' and self.args.n_train_examples > 0:
            n_examples = int(self.args.n_train_examples)
        elif split is 'valid' and self.args.n_valid_examples > 0:
            n_examples = int(self.args.n_valid_examples)
        elif split is 'test' and self.args.n_test_examples > 0:
            n_examples = int(self.args.n_test_examples)
        else:
            n_examples = None

        dataset = TripletDataset(text_data, annotation_data, self.args.k_negative, len(self.entity_dictionary), self.dictionary, n_examples)
        
        self.datasets[split] = dataset
