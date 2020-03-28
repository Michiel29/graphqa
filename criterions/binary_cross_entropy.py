import math

import numpy as np
import torch
import torch.nn.functional as F

from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('binary_cross_entropy_custom')
class BinaryCrossEntropy(FairseqCriterion):

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

        logits = model(sample)
        target = sample['target'].float()

        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='sum' if reduce else 'none')

        probs = torch.sigmoid(logits)
        predicted_class = probs >= 0.5

        sample_size = target.numel()
        logging_output = {
            'sample_size': sample_size,
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'ntokens_AB': sample['ntokens_AB'],
            'ntokens_mem': torch.numel(sample['textA']) + sample['textB_size'],
        }

        if self.args.eval_metric == 'accuracy':
            accuracy = (predicted_class == target).float().sum()
            logging_output['accuracy'] = utils.item(accuracy.data)
        else:
            raise NotImplementedError

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split, prefix='') -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get(prefix + 'sample_size', 0) for log in logging_outputs)
        weight = 0 if split == 'train' else sample_size

        loss_sum = sum(log.get(prefix + 'loss', 0) for log in logging_outputs)
        ntokens = sum(log.get(prefix + 'ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get(prefix + 'nsentences', 0) for log in logging_outputs)
        ntokens_AB = sum(log.get(prefix + 'ntokens_AB', 0) for log in logging_outputs)
        ntokens_mem = sum(log.get(prefix + 'ntokens_mem', 0) for log in logging_outputs)

        metrics.log_scalar(prefix + 'loss', loss_sum / sample_size, weight, round=3)
        metrics.log_scalar(prefix + 'wpb_mem', ntokens_mem, weight, round=3)
        metrics.log_scalar(prefix + 'wpb_AB', ntokens_AB, weight, round=3)

        accuracy_sum = sum(log.get(prefix + 'accuracy', 0) for log in logging_outputs)
        metrics.log_scalar(prefix + 'accuracy', accuracy_sum / sample_size, weight, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
