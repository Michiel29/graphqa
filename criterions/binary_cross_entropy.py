import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion

@register_criterion('binary_cross_entropy_custom')
class BinaryCrossEntropy(FairseqCriterion):

    def __init__(self, args, task):
        _Loss.__init__(self)
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

        logits = model(sample)
        target = sample['target'].float()
        
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='sum' if reduce else 'none')

        probs = torch.sigmoid(logits)
        predicted_class = probs >= 0.5
        accuracy = (predicted_class == target).float().mean()
        
        sample_size = target.numel()
        logging_output = {
            'sample_size': sample_size,
            'loss': utils.item(loss.data) if reduce else loss.data,
            'accuracy': utils.item(accuracy.data),
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'ntokens_AB': sample['ntokens_AB'],
            'ntokens_mem': torch.numel(sample['mentionA']) + sample['mentionB_size'],
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        accuracy_sum = sum(log.get('accuracy', 0) * log.get('sample_size', 0) for log in logging_outputs)
        
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        ntokens_AB = sum(log.get('ntokens_AB', 0) for log in logging_outputs)
        ntokens_mem = sum(log.get('ntokens_mem', 0) for log in logging_outputs)
        
        metrics.log_scalar('acc', accuracy_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('wpb_mem', ntokens_mem, sample_size, round=3)
        metrics.log_scalar('wpb_AB', ntokens_AB, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
