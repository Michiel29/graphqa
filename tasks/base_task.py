import logging
import warnings

from fairseq import metrics, utils
from fairseq.tasks import FairseqTask

logger = logging.getLogger(__name__)


class BaseTask(FairseqTask):
    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args)
        self.seed = args.seed
        self.dictionary = dictionary
        self.entity_dictionary = entity_dictionary
        self.mask_type = args.mask_type

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

        criterion.__class__.reduce_metrics(logging_outputs)

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary