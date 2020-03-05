import logging
import os

import numpy as np
import torch

from fairseq.data import (
    data_utils,
    iterators,
    FairseqDataset,
    Dictionary
)

from utils.data_utils import CustomDictionary, EntityDictionary
from datasets import GraphDataset
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
        parser.add_argument('--mask-type', default='head_tail', type=str,
                            help='method for masking entities in a sentence')
        parser.add_argument('--assign-head-tail', default='random', type=str,
                            help='method for assigning head and tail entities in a sentence')

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        raise NotImplementedError

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
