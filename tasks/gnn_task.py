from fairseq.data import FairseqDataset, iterators
from fairseq.tasks import register_task

import logging
import numpy as np
import os
import time

from datasets import (
    AnnotatedText,
    EpochSplitDataset,
    FixedSizeDataset,
    GraphDataset,
    PrependTokenDataset,
    GNNDataset,
)
from utils.data_utils import numpy_seed, safe_load_indexed_dataset
from utils.numpy_utils import MMapNumpyArray
from tasks import BaseTask

logger = logging.getLogger(__name__)


@register_task('gnn')
class GNNTask(BaseTask):

    NUM_SAMPLES_TO_COMPUTE_SAMPLE_SIZE = 20

    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args, dictionary, entity_dictionary)
        self.sample_sizes_mean = None

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data-path', help='path to data')
        parser.add_argument('--subsampling-strategy', type=str, default=None)
        parser.add_argument('--subsampling-cap', type=int, default=None)
        parser.add_argument('--mask-type', default='head_tail', type=str,
                            help='method for masking entities in a sentence')
        parser.add_argument('--non-mask-rate', default=1.0, type=float,
                            help='probability of not masking the entity with a [BLANK] token')
        parser.add_argument('--min-common-neighbors', type=int, default=None)
        parser.add_argument('--max-common-neighbors', type=int, default=None)
        parser.add_argument('--min-common-neighbors-for-the-last-edge', type=int, default=1)
        parser.add_argument('--num-text-chunks', type=int, default=None)
        parser.add_argument('--max-entities-size', type=int, default=None)
        parser.add_argument('--max-entities-from-queue', type=int, default=None)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        text_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.text'),
        )
        annotation_data = MMapNumpyArray(
            os.path.join(self.args.data_path, split + '.annotations.npy')
        )
        annotated_text = AnnotatedText(
            text_data=text_data,
            annotation_data=annotation_data,
            dictionary=self.dictionary,
            mask_type=self.args.mask_type,
            non_mask_rate=self.args.non_mask_rate,
        )

        graph_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.graph'),
        )
        graph = GraphDataset(
            edges=graph_data,
            subsampling_strategy=self.args.subsampling_strategy,
            subsampling_cap=self.args.subsampling_cap,
            seed=self.args.seed,
        )

        dataset = GNNDataset(
            annotated_text=annotated_text,
            graph=graph,
            dictionary=self.dictionary,
            min_common_neighbors=self.args.min_common_neighbors,
            max_common_neighbors=self.args.max_common_neighbors,
            min_common_neighbors_for_the_last_edge=self.args.min_common_neighbors_for_the_last_edge,
            max_entities_size=self.args.max_entities_size,
            max_entities_from_queue=self.args.max_entities_from_queue,
            max_tokens=self.args.max_tokens - 1, # for bos
            max_sentences=self.args.max_sentences,
            num_text_chunks=self.args.num_text_chunks,
            seed=self.args.seed,
        )
        if split == 'train' and self.args.epoch_size is not None:
            dataset = EpochSplitDataset(
                dataset=dataset,
                epoch_size=self.args.epoch_size,
                seed=self.args.seed,
            )

        dataset = PrependTokenDataset(dataset, self.dictionary.bos(), 'text')

        n_examples = getattr(self.args, 'n_' + split + '_examples', None)
        if n_examples is not None:
            dataset = FixedSizeDataset(
                dataset=dataset,
                size=n_examples,
                seed=self.seed,
            )

        self.datasets[split] = dataset

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
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

        # initialize the dataset with the correct starting epoch
        global_start_time = time.time()
        dataset.set_epoch(epoch)
        set_epoch_time = time.time() - global_start_time

        # get indices ordered by example size
        start_time = time.time()
        with numpy_seed('R3LTask', seed, epoch):
            indices = dataset.ordered_indices()
        sort_time = time.time() - start_time

        # create mini-batches with given size constraints
        start_time = time.time()
        batch_sampler = np.expand_dims(indices, 1)
        batch_by_size_time = time.time() - start_time
        logger.info(
            'get batch iterator [seed=%d, epoch=%d, num_shards=%d] is done in %.3f seconds '
            '(set epoch=%.3f, sorting=%.3f, batch by size=%.3f)' % (
                seed,
                epoch,
                num_shards,
                time.time() - global_start_time,
                set_epoch_time,
                sort_time,
                batch_by_size_time,
        ))

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

    def get_sample_size(self, batch, sizes):
        if self.sample_sizes_mean is None:
            assert len(self.datasets['train']) >= self.NUM_SAMPLES_TO_COMPUTE_SAMPLE_SIZE
            sample_sizes = []
            for i in range(self.NUM_SAMPLES_TO_COMPUTE_SAMPLE_SIZE):
                current_sample_size = len(self.datasets['train'][i]['target_text_idx'])
                sample_sizes.append(current_sample_size)
            self.sample_sizes_mean = float(sum(sample_sizes)) / len(sample_sizes)
        return int(self.sample_sizes_mean * len(batch))
