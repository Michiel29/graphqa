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
        super().__init__(args, task)
        self.head_idx = task.target_dictionary.head() if task.target_dictionary is not None else -100
        self.tail_idx = task.target_dictionary.tail() if task.target_dictionary is not None else -100


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

        logits = model(**sample['net_input'], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits])

        if sample_size != 0:
            targets = targets[masked_tokens]

        head_tail_mask = ((targets == self.head_idx) | (targets == self.tail_idx)).float()
        words_mask = 1 - head_tail_mask

        predicted_class = torch.argmax(logits, dim=1)
        accuracy_mask = (predicted_class == targets).float()

        accuracy_ht = 100 * accuracy_mask.dot(head_tail_mask)
        accuracy_w = 100 * accuracy_mask.dot(words_mask)
        accuracy = 100 * accuracy_mask.sum()

        # TODO: This might need to be modified for FP16 training
        # See https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/masked_lm.py
        # where they explicitly calculate loss in FP32
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets,
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
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        sample_size_ht = utils.item(sum(log.get('sample_size_ht', 0) for log in logging_outputs))
        sample_size_w = utils.item(sum(log.get('sample_size_w', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        accuracy = sum(log.get('accuracy', 0) for log in logging_outputs)
        accuracy_ht = sum(log.get('accuracy_ht', 0) for log in logging_outputs)
        accuracy_w = sum(log.get('accuracy_w', 0) for log in logging_outputs)

        accuracy = accuracy / sample_size if sample_size > 0 else 0
        accuracy_ht = accuracy_ht / sample_size_ht if sample_size_ht > 0 else 0
        accuracy_w = accuracy_w / sample_size_w if sample_size_w > 0 else 0

        accuracy = accuracy.round() if isinstance(accuracy, torch.Tensor) else round(accuracy, 3)
        accuracy_ht = accuracy_ht.round() if isinstance(accuracy_ht, torch.Tensor) else round(accuracy_ht, 3)
        accuracy_w = accuracy_w.round() if isinstance(accuracy_w, torch.Tensor) else round(accuracy_w, 3)

        # Hack round because "round" in log_scalar is broken
        metrics.log_scalar('acc', accuracy, 0, round=3)
        metrics.log_scalar('acc_ht', accuracy_ht, 0, round=3)
        metrics.log_scalar('acc_w', accuracy_w, 0, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True