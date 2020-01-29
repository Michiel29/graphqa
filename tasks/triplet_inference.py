import logging
import os
from collections import namedtuple
import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    iterators,
    FairseqDataset,
    PrependDataset
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


        dataset = TripletDataset(text_data, annotation_data, self.args.k_negative, len(self.entity_dictionary))
        
        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)  (do we need this?)
        # dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        self.datasets[split] = dataset
