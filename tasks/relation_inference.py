import logging
import os

import numpy as np

from utils.data_utils import safe_load_indexed_dataset
from datasets import GraphDataset
from tasks import BaseTask


logger = logging.getLogger(__name__)


def eval_str_list(x, type, length):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    assert len(x) == length
    return list(map(type, x))


class RelationInferenceTask(BaseTask):
    """Task for training inference models."""
    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args, dictionary, entity_dictionary)

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data-path', help='path to data')
        parser.add_argument('--subsampling-strategy', type=str, default=None)
        parser.add_argument('--subsampling-cap', type=int, default=None)
        parser.add_argument('--k-negative', default=1, type=int,
                            help='number of negative samples per mention')
        parser.add_argument('--negative-split-probs', default=None,
                            metavar='N1,N2,...,N_K',
                            type=lambda uf: eval_str_list(uf, type=float, length=3))
        parser.add_argument('--mask-type', default='head_tail', type=str,
                            help='method for masking entities in a sentence')
        parser.add_argument('--non-mask-rate', default=1.0, type=float,
                            help='probability of not masking the entity with a [BLANK] token')

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        raise NotImplementedError
