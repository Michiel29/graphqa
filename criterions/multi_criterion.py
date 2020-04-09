import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('multi_criterion')
class MultiCriterion(FairseqCriterion):

    def __init__(self, criterion_dict, weight_dict, sample_size, task):
        super().__init__(task)
        self.criterion_dict = nn.ModuleDict(criterion_dict)
        self.weight_dict = weight_dict
        self.sample_size = sample_size

    @classmethod
    def build_criterion(cls, args, task):
        raise Exception()

    def forward(self, model, sample, reduce=True):
        total_loss = 0.0
        total_logging_output = {}
        for task_name, criterion in self.criterion_dict.items():
            if sample[task_name] is not None:
                total_sample_size = sample[task_name]['sample_size']
                loss, _, logging_output = criterion(model, sample[task_name], reduce=reduce)
                total_loss += self.weight_dict[task_name] * loss / total_sample_size
                for k, v in logging_output.items():
                    total_logging_output[task_name + '_' + k] = v
        total_logging_output['loss'] = total_loss
        return total_loss, self.sample_size, total_logging_output

    def reduce_metrics(self, logging_outputs, split) -> None:
        """Aggregate logging outputs from data parallel training and update_freq."""
        weight = 0 if split == 'train' else len(logging_outputs)
        loss = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        metrics.log_scalar('loss', loss, weight, round=1)

        for task_name, criterion in self.criterion_dict.items():
            criterion.reduce_metrics(logging_outputs, split=split, prefix=task_name + '_')

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
