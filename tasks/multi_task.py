import logging
import os
import warnings

from fairseq import metrics, utils
from fairseq.tasks import register_task, TASK_REGISTRY

from tasks import BaseTask
from utils.data_utils import CustomDictionary, EntityDictionary


logger = logging.getLogger(__name__)


class ListTaskIterator(object):
    def _start_task_iterator(self, task_name):
        return self.tasks[task_name].get_batch_iterator(
            self.dataset_dict[task_name],
            max_tokens=self.args['tasks'][task_name]['max_tokens'],
            max_sentences=self.args['tasks'][task_name]['max_sentences'],
            max_positions=self.max_positions,
            ignore_invalid_inputs=self.ignore_invalid_inputs,
            required_batch_size_multiple=self.required_batch_size_multiple,
            seed=self.seed,
            num_shards=self.num_shards,
            shard_id=self.shard_id,
            num_workers=self.num_workers,
            epoch=self.epoch,
        ).next_epoch_itr(shuffle=self.shuffle, fix_batches_to_gpus=False)

    def __init__(
        self,
        args,
        tasks,
        dataset_dict,
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
        self.dataset_dict = dataset_dict
        self.max_positions = max_positions
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.required_batch_size_multiple = required_batch_size_multiple
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers
        self.epoch = epoch
        self.shuffle = shuffle

        self.iterators = {
            task_name: self._start_task_iterator(task_name)
            for task_name in self.tasks.keys()
        }
        self.count = start
        self.length = max([len(iterator) for iterator in self.iterators.values()])

    def __iter__(self):
        while self.count < self.length:
            result = {}
            for task_name in self.iterators.items():
                if not self.iterators[task_name].has_next():
                    self.iterators[task_name] = self._start_task_iterator(task_name)
                result[task_name] = next(self.iterators[task_name])
            yield result
            self.count += 1


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

    def state_dict(self):
        raise Exception()

    def load_state_dict(self, state_dict):
        raise Exception()


@register_task('multi_task')
class MultiTask(BaseTask):
    def __init__(self, args, dictionary, entity_dictionary, tasks):
        super().__init__(args, dictionary, entity_dictionary)
        self.tasks = tasks
        self.dataset_dict = {}

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
        if split not in self.dataset_dict:
            self.dataset_dict[split] = {}
        for task_name, task in self.tasks.items():
            task.load_dataset(split=split, epoch=epoch, combine=combine, **kwargs)
            self.dataset_dict[split][task_name] = task.dataset[split]

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        return ListTaskIteratorFactory(
            args=self.args,
            tasks=self.tasks,
            dataset_dict=dataset,
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
        from fairseq import criterions
        return criterions.build_criterion(args, self)

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, 'print_alignment', False),
            )

        from fairseq.sequence_generator import SequenceGenerator, SequenceGeneratorWithAlignment

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, 'sampling', False)
        sampling_topk = getattr(args, 'sampling_topk', -1)
        sampling_topp = getattr(args, 'sampling_topp', -1.0)
        diverse_beam_groups = getattr(args, 'diverse_beam_groups', -1)
        diverse_beam_strength = getattr(args, 'diverse_beam_strength', 0.5),
        match_source_len = getattr(args, 'match_source_len', False)
        diversity_rate = getattr(args, 'diversity_rate', -1)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError('Provided Search parameters are mutually exclusive.')
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'

        if sampling:
            search_strategy = search.Sampling(self.target_dictionary, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(self.target_dictionary, diversity_rate)
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        if getattr(args, 'print_alignment', False):
            seq_gen_cls = SequenceGeneratorWithAlignment
        else:
            seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            temperature=getattr(args, 'temperature', 1.),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            search_strategy=search_strategy,
        )

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        pass

    def update_step(self, num_updates):
        """Task level update when number of updates increases.

        This is called after the optimization step and learning rate
        update at each iteration.
        """
        pass

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        """[deprecated] Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            'The aggregate_logging_outputs API is deprecated. '
            'Please use the reduce_metrics API instead.'
        )
        with metrics.aggregate() as agg:
            self.reduce_metrics(logging_outputs, criterion)
            return agg.get_smoothed_values()

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        # backward compatibility for tasks that override aggregate_logging_outputs
        base_func = FairseqTask.aggregate_logging_outputs
        self_func = getattr(self, 'aggregate_logging_outputs').__func__
        if self_func is not base_func:
            utils.deprecation_warning(
                'Tasks should implement the reduce_metrics API. '
                'Falling back to deprecated aggregate_logging_outputs API.'
            )
            agg_logging_outputs = self.aggregate_logging_outputs(logging_outputs, criterion)
            for k, v in agg_logging_outputs.items():
                metrics.log_scalar(k, v)
            return

        if not any('ntokens' in log for log in logging_outputs):
            warnings.warn('ntokens not found in Criterion logging outputs, cannot log wpb or wps')
        else:
            ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
            metrics.log_scalar('wpb', ntokens, priority=180, round=1)
            metrics.log_speed('wps', ntokens, priority=90, round=1)

        if not any('nsentences' in log for log in logging_outputs):
            warnings.warn('nsentences not found in Criterion logging outputs, cannot log bsz')
        else:
            nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
            metrics.log_scalar('bsz', nsentences, priority=190, round=1)

        criterion.__class__.reduce_metrics(logging_outputs)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError