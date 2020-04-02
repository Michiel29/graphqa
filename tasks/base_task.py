import logging
import os
import time
import warnings

from fairseq import metrics, utils
from fairseq.data import (
    data_utils,
    FairseqDataset,
    iterators,
)
from fairseq.tasks import FairseqTask

from utils.dictionary import CustomDictionary, EntityDictionary


logger = logging.getLogger(__name__)


class BaseTask(FairseqTask):
    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args)
        self.args = args
        self.seed = args.seed
        self.dictionary = dictionary
        self.entity_dictionary = entity_dictionary
        self.mask_type = args.mask_type

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = CustomDictionary.load(dict_path)

        entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
        entity_dictionary = EntityDictionary.load(entity_dict_path)

        logger.info('dictionary: {} types'.format(len(dictionary)))
        logger.info('entity dictionary: {} types'.format(len(entity_dictionary)))

        task = cls(args, dictionary, entity_dictionary)
        return task

    def reduce_metrics(self, logging_outputs, criterion):
        if not any('ntokens' in log for log in logging_outputs):
            warnings.warn('ntokens not found in Criterion logging outputs, cannot log wpb or wps')
        else:
            ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
            metrics.log_scalar('wpb', ntokens, priority=180, round=1)
            # TODO(urikz): Latest version of fairseq also has additional argument "ignore_first"
            metrics.log_speed('wps', ntokens, priority=90, round=1)

        if not any('nsentences' in log for log in logging_outputs):
            warnings.warn('nsentences not found in Criterion logging outputs, cannot log bsz')
        else:
            nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
            metrics.log_scalar('ns', nsentences, priority=190, round=1)

        if not any('sample_size' in log for log in logging_outputs):
            warnings.warn('sample_size not found in Criterion logging outputs, cannot log bsz')
        else:
            sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
            metrics.log_scalar('bsz', sample_size, priority=190, round=1)

        criterion.__class__.reduce_metrics(logging_outputs, self)

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

        logger.info('get batch iterator: START (seed=%d, epoch=%d, num_shards=%d)' % (
            seed,
            epoch,
            num_shards,
        ))

        # initialize the dataset with the correct starting epoch
        start_time = time.time()
        dataset.set_epoch(epoch)
        logger.info('get batch iterator: set epoch (seed=%d, epoch=%d) is done in %.3f seconds' % (
            seed,
            epoch,
            time.time() - start_time,
        ))

        # get indices ordered by example size
        start_time = time.time()
        with data_utils.numpy_seed(seed, epoch):
            indices = dataset.ordered_indices()
        logger.info('get batch iterator: sorting (seed=%d, epoch=%d) is done in %.3f seconds' % (
            seed,
            epoch,
            time.time() - start_time,
        ))

        # create mini-batches with given size constraints
        start_time = time.time()
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
        logger.info('get batch iterator: batch by size is done in %.3f seconds' % (
            time.time() - start_time,
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


    # def train_step(
    #     self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    # ):
    #     """
    #     Do forward and backward, and return the loss as computed by *criterion*
    #     for the given *model* and *sample*.
    #     Args:
    #         sample (dict): the mini-batch. The format is defined by the
    #             :class:`~fairseq.data.FairseqDataset`.
    #         model (~fairseq.models.BaseFairseqModel): the model
    #         criterion (~fairseq.criterions.FairseqCriterion): the criterion
    #         optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
    #         update_num (int): the current update
    #         ignore_grad (bool): multiply loss by 0 if this is set to True
    #     Returns:
    #         tuple:
    #             - the loss
    #             - the sample size, which is used as the denominator for the
    #               gradient
    #             - logging outputs to display while training
    #     """
    #     model.train()
    #     model.set_num_updates(update_num)
    #     loss, sample_size, logging_output = criterion(model, sample)
    #     if ignore_grad:
    #         loss *= 0
    #     if not isinstance(loss, float):
    #         optimizer.backward(loss)
    #     return loss, sample_size, logging_output

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def reporter(self, pred, target, logging_output):
        return logging_output