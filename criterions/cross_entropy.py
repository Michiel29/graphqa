import numpy as np
import torch
import torch.nn.functional as F

from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion

from utils.diagnostic_utils import Diagnostic


@register_criterion('cross_entropy_custom')
class CrossEntropy(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task)
        self.args = args
        self.task = task

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    @classmethod
    def build_criterion(cls, args, task):
        return cls(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        model_output = model(sample)
        target = sample['target']

        # diag = Diagnostic(self.task.dictionary, self.task.entity_dictionary, self.task)
        # diag.inspect_batch(sample, scores=model_output)

        loss = F.cross_entropy(model_output, target, reduction='sum' if reduce else 'none')
        pred = torch.argmax(model_output, dim=1)

        sample_size = target.numel()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'sample_size': sample_size,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'accuracy': utils.item((pred == target).float().sum()),
        }

        if 'ntokens_AB' in sample.keys():
            logging_output['ntokens_AB'] = sample['ntokens_AB']
        if 'ntokens_mem' in sample.keys():
            logging_output['ntokens_mem'] = sample['ntokens_mem']
        if 'yield' in sample.keys():
            logging_output['yield'] = sample['yield']
        if 'rel_cov' in sample.keys():
            logging_output['rel_cov'] = sample['rel_cov']


        logging_output = self.task.reporter(target, pred, logging_output)

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split, prefix='') -> None:
        """Aggregate logging outputs from data parallel training and update_freq."""
        sample_size = sum(log.get(prefix + 'sample_size', 0) for log in logging_outputs)
        weight = 0 if split == 'train' else sample_size

        loss_sum = sum(log.get(prefix + 'loss', 0) for log in logging_outputs)
        accuracy_sum = sum(log.get(prefix + 'accuracy', 0) for log in logging_outputs)

        metrics.log_scalar(prefix + 'loss', loss_sum / sample_size, weight, priority=0, round=3)
        metrics.log_scalar(prefix + 'acc', accuracy_sum / sample_size, weight, priority=10, round=3)
        if split == 'train':
            metrics.log_scalar(
                prefix + 'acc_avg',
                accuracy_sum / sample_size,
                sample_size,
                priority=10,
                round=3,
            )

        if 'yield' in logging_outputs[0].keys():
            yield_pct = np.array([
                utils.item(x) for x in [log.get('yield') for log in logging_outputs] if x is not None
            ])
            metrics.log_scalar(prefix + 'yield', yield_pct.mean(), priority=100, round=3)
        if 'rel_cov' in logging_outputs[0].keys():
            rel_cov = np.array([
                utils.item(x) for x in [log.get('rel_cov') for log in logging_outputs] if x is not None
            ])
            metrics.log_scalar(prefix + 'rel_cov', rel_cov.mean(), priority=100, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
