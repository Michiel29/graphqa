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
from fairseq.tasks import FairseqTask, register_task
from datasets.fewrel_dataset import FewRelDataset

from utils.data_utils import CustomDictionary

logger = logging.getLogger(__name__)

@register_task('fewrel')
class FewRelTask(FairseqTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        
        """Required either in config or cl"""
        parser.add_argument('--data_path', help='path to data')
        parser.add_argument('--n_way', default=5, help='number of few-shot classes')
        parser.add_argument('--n_shot', default=1, help='number of few-shot examples')

        
        """Optional"""
        # optional arguments here
     

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

        #  temp for testing remove 
        self.entity_dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = CustomDictionary.load(dict_path)
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

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

        
        dataset = FewRelDataset(text_data, annotation_data, relation_data, self.dictionary, self.args.n_way, self.args.n_shot, self.args.dataset_size)

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
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
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