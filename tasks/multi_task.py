import logging
import numpy as np
import os
import warnings

from fairseq import metrics, utils

from fairseq.data import data_utils
from fairseq.tasks import register_task, TASK_REGISTRY

from tasks import BaseTask
from utils.data_utils import CustomDictionary, EntityDictionary


logger = logging.getLogger(__name__)


class ListTaskIterator(object):
    def _start_task_iterator(self, task_name):
        epoch_iterator = self.tasks[task_name].get_batch_iterator(
            self.datasets[task_name],
            max_tokens=self.args.tasks[task_name]['max_tokens'],
            max_sentences=self.args.tasks[task_name]['max_sentences'],
            max_positions=self.max_positions,
            ignore_invalid_inputs=self.ignore_invalid_inputs,
            required_batch_size_multiple=self.required_batch_size_multiple,
            seed=self.seed,
            num_shards=self.num_shards,
            shard_id=self.shard_id,
            num_workers=self.num_workers,
            epoch=self.epoch,
        )
        l = list(epoch_iterator.frozen_batches)
        if self.shuffle:
            with data_utils.numpy_seed(self.seed, self.epoch):
                np.random.shuffle(l)
                epoch_iterator.frozen_batches = tuple(l)
        new_sample_sizes = np.array([len(x) for x in l]).reshape([-1, self.num_shards]).sum(axis=-1)
        if task_name not in self.sample_sizes:
            self.sample_sizes[task_name] = new_sample_sizes
        else:
            self.sample_sizes[task_name] = np.concatenate([self.sample_sizes[task_name], new_sample_sizes])

        return epoch_iterator.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)

    def __init__(
        self,
        args,
        tasks,
        datasets,
        max_positions,
        ignore_invalid_inputs,
        required_batch_size_multiple,
        seed,
        num_shards,
        shard_id,
        num_workers,
        epoch,
        shuffle,
        start=0,
    ):
        self.args = args
        self.tasks = tasks
        self.datasets = datasets
        self.max_positions = max_positions
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.required_batch_size_multiple = required_batch_size_multiple
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers
        self.epoch = epoch
        self.shuffle = shuffle
        self.sample_sizes = {}

        self.iterators = {
            task_name: self._start_task_iterator(task_name)
            for task_name in self.tasks.keys()
        }
        self.count = start
        self.length = max([len(iterator) for iterator in self.iterators.values()])
        self.itr = iter(self)

    def __iter__(self):
        while self.count < self.length:
            result = {}
            for task_name in self.iterators.keys():
                if not self.iterators[task_name].has_next():
                    self.iterators[task_name] = self._start_task_iterator(task_name)
                result[task_name] = next(self.iterators[task_name])
                result[task_name]['sample_size'] = self.sample_sizes[task_name][self.count]
            self.count += 1
            yield result

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        return self.count < len(self)

    def __len__(self):
        return self.length


class ListTaskIteratorFactory(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.epoch = kwargs['epoch']
        self._cur_epoch_itr = None
        self._next_epoch_itr = None

    def __len__(self):
        return len(self._cur_epoch_itr)

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        self._cur_epoch_itr = ListTaskIterator(shuffle=shuffle, **self.kwargs)
        self._next_epoch_itr = None
        return self._cur_epoch_itr


    def state_dict(self):
        return {
            'epoch': self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._cur_epoch_itr.has_next()


@register_task('multi_task')
class MultiTask(BaseTask):
    def __init__(self, args, dictionary, entity_dictionary, tasks):
        super().__init__(args, dictionary, entity_dictionary)
        self.tasks = tasks
        self.datasets = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = CustomDictionary.load(dict_path)

        entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
        entity_dictionary = EntityDictionary.load(entity_dict_path)

        logger.info('dictionary: {} types'.format(len(dictionary)))
        logger.info('entity dictionary: {} types'.format(len(entity_dictionary)))

        tasks = {
            task_name: TASK_REGISTRY[task_name](args, dictionary, entity_dictionary)
            for task_name in args.tasks
        }
        task = cls(args, dictionary, entity_dictionary, tasks)
        return task

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        if split not in self.datasets:
            self.datasets[split] = {}
        for task_name, task in self.tasks.items():
            task.load_dataset(split=split, epoch=epoch, combine=combine, **kwargs)
            self.datasets[split][task_name] = task.dataset(split)

    def dataset(self, split):
        return self.datasets[split]

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        return ListTaskIteratorFactory(
            args=self.args,
            tasks=self.tasks,
            datasets=dataset,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )

    def build_criterion(self, args):
        from fairseq.criterions import CRITERION_REGISTRY
        import criterions
        cc, cc_weights = {}, {}
        for task_name, task_args in args.tasks.items():
            cc[task_name] = CRITERION_REGISTRY[task_args['criterion']].build_criterion(args, self.tasks[task_name])
            cc_weights[task_name] = task_args['weight']
        return criterions.MultiCriterion(cc, cc_weights, self)

    def reduce_metrics(self, logging_outputs, criterion):
        assert len(logging_outputs) == 1
        for k, v in logging_outputs[0].items():
            if 'ntokens' in k:
                metrics.log_scalar(k[:-len('ntokens')] + 'wpb', v, priority=180, round=1)
                # TODO(urikz): Latest version of fairseq also has additional argument "ignore_first"
                metrics.log_speed(k[:-len('ntokens')] + 'wps', v, priority=90, round=1)
            elif 'nsentences' in k:
                metrics.log_scalar(k[:-len('nsentences')] + 'ns', v, priority=190, round=1)
            elif 'sample_size' in k:
                metrics.log_scalar(k[:-len('sample_size')] + 'bsz', v, priority=190, round=1)
        criterion.reduce_metrics(logging_outputs)