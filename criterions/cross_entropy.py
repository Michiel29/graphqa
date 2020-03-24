
import math

import numpy as np
import torch
import torch.nn.functional as F

from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion

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

        split = sample['split']
        target = sample['target']
        model_output = model(sample)
        
        loss = F.cross_entropy(model_output, target, reduction='sum' if reduce else 'none')
        predicted_class = torch.argmax(model_output, dim=1)

        sample_size = target.numel()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'sample_size': sample_size,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
        }

        if self.args.eval_metric == 'accuracy':
            accuracy = (predicted_class == target).float().sum()
            logging_output['accuracy'] = utils.item(accuracy.data)
        elif self.args.eval_metric == 'f1':
            self.task.eval_metrics[split].update_metrics(predicted_class, target)
            logging_output['f1'] = self.task.eval_metrics[split].f1_value
        else:
            raise NotImplementedError

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, eval_metric='accuracy', prefix='') -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = sum(log.get(prefix + 'loss', 0) for log in logging_outputs)
        sample_size = sum(log.get(prefix + 'sample_size', 0) for log in logging_outputs)

        ntokens = sum(log.get(prefix + 'ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get(prefix + 'nsentences', 0) for log in logging_outputs)

        metrics.log_scalar(prefix + 'loss', loss_sum / sample_size, sample_size, round=3)

        if eval_metric == 'accuracy':
            accuracy_sum = sum(log.get(prefix + 'accuracy', 0) for log in logging_outputs)
            metrics.log_scalar(prefix + 'accuracy', accuracy_sum / sample_size, sample_size, round=3)
        elif eval_metric == 'f1':
            metrics.log_scalar(prefix + 'f1', logging_outputs[-1].get(prefix + 'f1', 0), sample_size, round=3)
        else:
            raise NotImplementedError

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
