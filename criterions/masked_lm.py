import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('masked_lm_custom')
class MaskedLmCustomLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """
    def __init__(self, args, task):
        super().__init__(task)
        self.head_idx = task.target_dictionary.head() if task.target_dictionary is not None else -100
        self.tail_idx = task.target_dictionary.tail() if task.target_dictionary is not None else -100
        self.padding_idx = task.target_dictionary.pad() if task.target_dictionary is not None else -100
        self.e1_start_idx = task.target_dictionary.e1_start() if task.target_dictionary is not None else -100
        self.e1_end_idx = task.target_dictionary.e1_end() if task.target_dictionary is not None else -100
        self.e2_start_idx = task.target_dictionary.e2_start() if task.target_dictionary is not None else -100
        self.e2_end_idx = task.target_dictionary.e2_end() if task.target_dictionary is not None else -100

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
        logits = encoder(**sample['net_input'], masked_tokens=masked_tokens, use_lm_head=True)[0]
        targets = model.get_targets(sample, [logits])

        if sample_size != 0:
            targets = targets[masked_tokens]

        head_tail_mask = (
            (targets == self.head_idx)
            | (targets == self.tail_idx)
            | (targets == self.e1_start_idx)
            | (targets == self.e1_end_idx)
            | (targets == self.e2_start_idx)
            | (targets == self.e2_end_idx)
        ).float()
        words_mask = 1 - head_tail_mask

        predicted_class = torch.argmax(logits, dim=1)
        accuracy_mask = (predicted_class == targets).float()

        accuracy_ht = 100 * accuracy_mask.dot(head_tail_mask)
        accuracy_w = 100 * accuracy_mask.dot(words_mask)
        accuracy = 100 * accuracy_mask.sum()

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
            'accuracy_ht': accuracy_ht,
            'accuracy_w': accuracy_w,
            'sample_size_ht': head_tail_mask.sum(),
            'sample_size_w': words_mask.sum(),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split, prefix='') -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(sum(log.get(prefix + 'sample_size', 0) for log in logging_outputs))
        weight = 0 if split == 'train' else sample_size

        loss_sum = utils.item(sum(log.get(prefix + 'loss', 0) for log in logging_outputs))
        sample_size_ht = utils.item(sum(log.get(prefix + 'sample_size_ht', 0) for log in logging_outputs))
        sample_size_w = utils.item(sum(log.get(prefix + 'sample_size_w', 0) for log in logging_outputs))

        metrics.log_scalar(prefix + 'loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived(prefix + 'ppl', lambda meters: utils.get_perplexity(meters[prefix + 'loss'].avg))

        accuracy = sum(log.get(prefix + 'accuracy', 0) for log in logging_outputs)
        accuracy_ht = sum(log.get(prefix + 'accuracy_ht', 0) for log in logging_outputs)
        accuracy_w = sum(log.get(prefix + 'accuracy_w', 0) for log in logging_outputs)

        accuracy = accuracy / sample_size if sample_size > 0 else 0
        accuracy_ht = accuracy_ht / sample_size_ht if sample_size_ht > 0 else 0
        accuracy_w = accuracy_w / sample_size_w if sample_size_w > 0 else 0

        accuracy = accuracy.round() if isinstance(accuracy, torch.Tensor) else round(accuracy, 3)
        accuracy_ht = accuracy_ht.round() if isinstance(accuracy_ht, torch.Tensor) else round(accuracy_ht, 3)
        accuracy_w = accuracy_w.round() if isinstance(accuracy_w, torch.Tensor) else round(accuracy_w, 3)

        # Hack round because "round" in log_scalar is broken
        metrics.log_scalar(prefix + 'acc', accuracy, weight, round=3)
        metrics.log_scalar(prefix + 'acc_ht', accuracy_ht, weight, round=3)
        metrics.log_scalar(prefix + 'acc_w', accuracy_w, weight, round=3)

        metrics.log_scalar(prefix + 'num_masked', sample_size, weight, round=3, priority=1e9)
        metrics.log_scalar(prefix + 'num_masked_ht', sample_size_ht, weight, round=3, priority=1e9)
        metrics.log_scalar(prefix + 'num_masked_w', sample_size_w, weight, round=3, priority=1e9)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True