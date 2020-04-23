import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('masked_lm_custom')
class MaskedLmCustomLoss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(task)
        self.task = task
        weights = torch.ones(len(task.dictionary))
        weights[task.dictionary.special_tokens()] = 0
        self.weights = nn.Parameter(weights, requires_grad=False)

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
        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None

        if hasattr(model, 'encoder'):
            encoder = model.encoder
        else:
            encoder = model
        logits = encoder(
            src_tokens=sample['src_tokens'],
            src_lengths=sample['src_lengths'],
            masked_tokens=masked_tokens,
            use_lm_head=True,
        )[0]
        targets = model.get_targets(sample, [logits])

        if sample_size != 0:
            targets = targets[masked_tokens]

        words_mask = self.weights[targets]
        special_tokens_mask = 1 - words_mask

        predicted_class = torch.argmax(logits, dim=1)
        accuracy_mask = (predicted_class == targets).float()

        accuracy_s = accuracy_mask.dot(special_tokens_mask)
        accuracy_w = accuracy_mask.dot(words_mask)
        accuracy = accuracy_mask.sum()

        loss = F.nll_loss(
            F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'accuracy': accuracy,
            'accuracy_s': accuracy_s,
            'accuracy_w': accuracy_w,
            'sample_size_s': special_tokens_mask.sum(),
            'sample_size_w': words_mask.sum(),
        }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split, prefix='') -> None:
        """Aggregate logging outputs from data parallel training and update_freq."""
        sample_size = utils.item(sum(log.get(prefix + 'sample_size', 0) for log in logging_outputs))
        sample_size_s = utils.item(sum(log.get(prefix + 'sample_size_s', 0) for log in logging_outputs))
        sample_size_w = utils.item(sum(log.get(prefix + 'sample_size_w', 0) for log in logging_outputs))

        weight = 0 if split == 'train' else sample_size
        weight_s = 0 if split == 'train' else sample_size_s
        weight_w = 0 if split == 'train' else sample_size_w

        loss_sum = utils.item(sum(log.get(prefix + 'loss', 0) for log in logging_outputs))

        metrics.log_scalar(
            prefix + 'loss',
            loss_sum / sample_size / math.log(2),
            weight,
            priority=0,
            round=3,
        )
        metrics.log_derived(
            prefix + 'ppl',
            lambda meters: utils.get_perplexity(meters[prefix + 'loss'].avg),
            priority=100,
        )

        accuracy = sum(log.get(prefix + 'accuracy', 0) for log in logging_outputs)
        accuracy_s = sum(log.get(prefix + 'accuracy_s', 0) for log in logging_outputs)
        accuracy_w = sum(log.get(prefix + 'accuracy_w', 0) for log in logging_outputs)

        accuracy = accuracy / sample_size if sample_size > 0 else 0
        accuracy_s = accuracy_s / sample_size_s if sample_size_s > 0 else 0
        accuracy_w = accuracy_w / sample_size_w if sample_size_w > 0 else 0

        metrics.log_scalar(prefix + 'acc', accuracy, weight, priority=100, round=3)
        metrics.log_scalar(prefix + 'acc_s', accuracy_s, weight_s, priority=10, round=3)
        metrics.log_scalar(prefix + 'acc_w', accuracy_w, weight_w, priority=10, round=3)

        metrics.log_scalar(prefix + 'num_masked', sample_size, weight, round=3, priority=1e9)
        metrics.log_scalar(prefix + 'num_masked_s', sample_size_s, weight_s, round=3, priority=1e9)
        metrics.log_scalar(prefix + 'num_masked_w', sample_size_w, weight_w, round=3, priority=1e9)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True