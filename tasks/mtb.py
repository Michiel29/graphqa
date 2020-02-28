import logging
import os

import numpy as np
import torch

from fairseq.data import (
    data_utils,
    iterators,
    FairseqDataset,
    PrependDataset,
    PrependTokenDataset,
    Dictionary
)
from fairseq.tasks import FairseqTask, register_task

from utils.data_utils import MTBDictionary, EntityDictionary
from tasks import RelationInferenceTask
from datasets import MTBDataset, FixedSizeDataset

logger = logging.getLogger(__name__)

@register_task('mtb')
class MTBTask(RelationInferenceTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data-path', help='path to data')
        parser.add_argument('--case0_prob', default=0.5, type=float,
                            help='probability of sampling a pair of sentences which share both head and tail entities')
        parser.add_argument('--case1_prob', default=0.35, type=float,
                            help='probability of sampling a pair of sentences which share only one entity')
        parser.add_argument('--n_tries', default=10, type=int,
                            help='number of attempts to sample mentions for a given case')
        parser.add_argument('--alpha', default=0.7, type=float,
                            help='probability of not masking the entity with a [BLANK] token')

        """Optional"""
        # optional arguments here

    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args, dictionary, entity_dictionary)
        self.entity_dictionary = entity_dictionary
        self.seed = args.seed
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = MTBDictionary.load(dict_path)

        entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
        entity_dictionary = EntityDictionary.load(entity_dict_path)

        logger.info('dictionary: {} types'.format(len(dictionary)))
        logger.info('entity dictionary: {} types'.format(len(entity_dictionary)))

        task = cls(args, dictionary, entity_dictionary)
        
        task.load_graph()

        return task

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        text_data, annotation_data = self.load_annotated_text(split)
        if split == 'valid':
            graph_text_data, graph_annotation_data = self.load_annotated_text('train')
        else:
            graph_text_data, graph_annotation_data = self.load_annotated_text('train') 

        self.datasets[split] = MTBDataset(
            split,
            text_data,
            annotation_data,
            self.graph,
            graph_text_data,
            graph_annotation_data,
            len(self.entity_dictionary),
            self.dictionary,
            self.args.max_positions,
            self.args.case0_prob,
            self.args.case1_prob,
            self.args.n_tries,
            shift_annotations=1, # because of the PrependTokenDataset
            alpha=self.args.alpha,
        )

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0):
        """ 
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """

        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices, dataset, max_positions-4, raise_exception=(not ignore_invalid_inputs),
            )   

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.size, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )   

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )   
        
        return epoch_iter
