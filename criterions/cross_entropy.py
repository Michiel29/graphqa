
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss



from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion

@register_criterion('cross_entropy_custom')
class CrossEntropy(FairseqCriterion):

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

        model_output = model(sample)
        target = torch.cuda.LongTensor(sample['target'])
        loss = F.cross_entropy(model_output, target, reduce=reduce)

        sample_size = target.numel()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'sample_size': sample_size,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'sample_size': sample_size,
        }

        return agg_output
