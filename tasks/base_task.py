import logging
import warnings

from fairseq import metrics, utils
from fairseq.tasks import FairseqTask

logger = logging.getLogger(__name__)


class BaseTask(FairseqTask):
    def __init__(self, args):
        super().__init__(args)

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        # backward compatibility for tasks that override aggregate_logging_outputs
        base_func = FairseqTask.aggregate_logging_outputs
        self_func = getattr(self, 'aggregate_logging_outputs').__func__
        if self_func is not base_func:
            raise Exception(
                'Tasks should implement the reduce_metrics API. '
                'Falling back to deprecated aggregate_logging_outputs API.'
            )

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

        criterion.__class__.reduce_metrics(logging_outputs)
