from fairseq.tasks import register_task
import logging
import os

from datasets import (
    AnnotatedText,
    EpochSplitDataset,
    FixedSizeDataset,
    GraphDataset,
    PrependTokenDataset,
    R3LDataset,
)
from utils.data_utils import safe_load_indexed_dataset
from utils.numpy_utils import MMapNumpyArray
from tasks import BaseTask

logger = logging.getLogger(__name__)


@register_task('r3l')
class R3LTask(BaseTask):
    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args, dictionary, entity_dictionary)

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
        parser.add_argument('--min-common-neighbors-for-the-last-edge', type=int, default=1)

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

        dataset = R3LDataset(
            annotated_text=annotated_text,
            graph=graph,
            min_common_neighbors=args.min_common_neighbors,
            min_common_neighbors_for_the_last_edge=args.min_common_neighbors_for_the_last_edge,
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
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
        with data_utils.numpy_seed(seed, epoch):
            indices = dataset.ordered_indices()
        sort_time = time.time() - start_time

        # create mini-batches with given size constraints
        start_time = time.time()
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
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