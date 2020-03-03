import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('masked_lm_custom')
class MultiCriterion(FairseqCriterion):

    def __init__(self, criterion_dict):
        super().__init__(None, None)
        self.criterion_dict = nn.ModuleDict(criterion_dict)

    @classmethod
    def build_criterion(cls, args, task):
        raise Exception()

    def forward(self, model, sample, reduce=True):
        total_loss = 0
        total_sample_size = {}
        total_logging_output = {}
        for task_name, criterion in self.criterion_dict.items():
            loss, sample_size, logging_output = criterion(model, sample[task_name], reduce=reduce)
            total_loss += loss # * weight
            total_sample_size[task_name] = sample_size
            for k, v in logging_output.items():
                total_logging_output[task_name + '_' + k] = v
        return total_loss, total_sample_size, total_logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            'Criterions should implement the reduce_metrics API. '
            'Falling back to deprecated aggregate_logging_outputs API.'
        )
        agg_logging_outputs = cls.aggregate_logging_outputs(logging_outputs)
        for k, v in agg_logging_outputs.items():
            if k in {'nsentences', 'ntokens', 'sample_size'}:
                continue
            metrics.log_scalar(k, v)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
