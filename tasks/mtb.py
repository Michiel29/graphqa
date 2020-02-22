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
from datasets import MTBDataset, FixedSizeDataset

#from cython_modules.construct_graph import construct_graph

logger = logging.getLogger(__name__)

@register_task('mtb')
class MTBTask(FairseqTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data-path', help='path to data')
        parser.add_argument('--case0_prob', default=0.5, type=float,
                            help='probability of sampling a pair of sentences which share both head and tail entities')
        parser.add_argument('--case1_prob', default=0.4, type=float,
                            help='probability of sampling a pair of sentences which share only one entity')
        parser.add_argument('--n_tries', default=10, type=int,
                            help='number of attempts to sample mentions for a given case')
        parser.add_argument('--alpha', default=0.7, type=float,
                            help='probability of not masking the entity with a [BLANK] token')

        """Optional"""
        # optional arguments here

    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args)
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
        task.load_dataset('train')

        # here only temporarily
        def construct_graph(annotation_data, n_entities):
            from collections import defaultdict
            from itertools import combinations
            neighbor_list = [defaultdict(int) for entity in range(n_entities)]
            edge_dict = defaultdict(list)

            for sentence_idx in range(len(annotation_data)):
                entity_ids = annotation_data[sentence_idx].reshape(-1, 3)[:, -1].numpy()

                for a, b in combinations(entity_ids, 2):
                    neighbor_list[a][b] += 1
                    neighbor_list[b][a] += 1

                    edge_dict[frozenset({a, b})].append(sentence_idx)

            return neighbor_list, edge_dict

        logger.info('beginning train graph construction')
        task.neighbor_list, task.edge_dict = construct_graph(task.datasets['train'].annotation_data, len(entity_dictionary))
        logger.info('finished train graph construction')

        task.datasets['train'].neighbor_list = task.neighbor_list
        task.datasets['train'].edge_dict = task.edge_dict

        return task

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        # Don't reload datasets that were already setup earlier
        if split in self.datasets:
            return

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

        dataset = MTBDataset(
            text_data,
            annotation_data,
            len(self.entity_dictionary),
            self.dictionary,
            self.args.case0_prob,
            self.args.case1_prob,
            self.args.max_positions,
            self.args.n_tries,
            self.args.alpha,
            shift_annotations=1, # because of the PrependTokenDataset
        )   

        self.datasets[split] = dataset
        
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
                indices, dataset, max_positions, raise_exception=(not ignore_invalid_inputs),
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

    @property
    def source_dictionary(self):
        return self.dictionary
