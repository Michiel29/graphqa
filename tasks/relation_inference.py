import logging
import os

import numpy as np
import torch

from fairseq.data import (
    data_utils,
    iterators,
    FairseqDataset,
    PrependTokenDataset,
    Dictionary
)

from utils.data_utils import CustomDictionary, EntityDictionary
from datasets import FixedSizeDataset, GraphDataset
from tasks import BaseTask

logger = logging.getLogger(__name__)


class RelationInferenceTask(BaseTask):
    """Task for training inference models."""
    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args, dictionary, entity_dictionary)
        self.load_graph()

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data-path', help='path to data')
        parser.add_argument('--k-negative', default=1, type=int,
                            help='number of negative samples per mention')

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        raise NotImplementedError

    def load_annotated_text(self, split):
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

        n_examples = int(getattr(self.args, 'n_' + split + '_examples', -1))

        text_data = FixedSizeDataset(text_data, n_examples)
        annotation_data = FixedSizeDataset(annotation_data, n_examples)

        return text_data, annotation_data

    def load_graph(self):
        neighbor_path = os.path.join(self.args.data_path, 'neighbors')
        edge_path = os.path.join(self.args.data_path, 'edges')

        neighbor_data = data_utils.load_indexed_dataset(
            neighbor_path,
            None,
            dataset_impl='mmap'
        )

        if neighbor_data is None:
            raise FileNotFoundError('Dataset (graph) not found: {}'.format(neighbor_path))

        edge_data = data_utils.load_indexed_dataset(
            edge_path,
            None,
            dataset_impl='mmap'
        )

        if edge_data is None:
            raise FileNotFoundError('Dataset (graph) not found: {}'.format(edge_path))

        self.graph = GraphDataset(neighbor_data, edge_data)
