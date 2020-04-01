import numpy as np
import collections
import torch
import torch.nn.functional as F

from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from utils.logging_utils import compute_confusion_matrix, MacroF1Meter, MicroF1Meter


@register_criterion('cross_entropy_custom')
class CrossEntropy(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task)
        self.args = args
        self.task = task
        self.num_classes = args.num_classes
        self.eval_metric = args.eval_metric

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

        target = sample['target']
        model_output = model(sample)

        loss = F.cross_entropy(model_output, target, reduction='sum' if reduce else 'none')
        pred = torch.argmax(model_output, dim=1)

        sample_size = target.numel()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'sample_size': sample_size,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'acc': utils.item((pred == target).float().sum()),
        }

        if self.eval_metric == 'macro_f1':
            fn, tp, fp = compute_confusion_matrix(target.cpu().numpy(), pred.detach().cpu().numpy(), 'macro', num_classes=self.num_classes)
            for i in fn.keys():
                logging_output['fn_' + str(i)] = fn[i]
                logging_output['tp_' + str(i)] = tp[i]
                logging_output['fp_' + str(i)] = fp[i]
        elif self.eval_metric == 'micro_f1':
            fn, tp, fp = compute_confusion_matrix(target.cpu().numpy(), pred.detach().cpu().numpy(), 'micro', num_classes=self.num_classes, task=self.args.task)
            logging_output['fn'] = fn
            logging_output['tp'] = tp
            logging_output['fp'] = fp

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, task, prefix='') -> None:
        """Aggregate logging outputs from data parallel training."""
        split = task.split
        task_name = task.args.task
        num_classes = task.args.num_classes
        eval_metric = task.args.eval_metric

        sample_size = sum(log.get(prefix + 'sample_size', 0) for log in logging_outputs)
        weight = 0 if split == 'train' else sample_size

        loss_sum = sum(log.get(prefix + 'loss', 0) for log in logging_outputs)
        ntokens = sum(log.get(prefix + 'ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get(prefix + 'nsentences', 0) for log in logging_outputs)

        metrics.log_scalar(prefix + 'loss', loss_sum / sample_size, weight, round=3)

        acc_sum = sum(log.get(prefix + 'acc', 0) for log in logging_outputs)
        metrics.log_scalar(prefix + 'acc', acc_sum / sample_size, weight, round=3)

        if eval_metric == 'macro_f1':
            fn = collections.defaultdict(int)
            tp = collections.defaultdict(int)
            fp = collections.defaultdict(int)
            for i in range(num_classes):
                fn[i] = sum(log.get(prefix + 'fn_' + str(i), 0) for log in logging_outputs)
                tp[i] = sum(log.get(prefix + 'tp_' + str(i), 0) for log in logging_outputs)
                fp[i] = sum(log.get(prefix + 'fp_' + str(i), 0) for log in logging_outputs)
            metrics.log_custom(MacroF1Meter, 'macro_f1', fn, tp, fp, task_name, split, weight)
        elif eval_metric == 'micro_f1':
            fn = sum(log.get(prefix + 'fn', 0) for log in logging_outputs)
            tp = sum(log.get(prefix + 'tp', 0) for log in logging_outputs)
            fp = sum(log.get(prefix + 'fp', 0) for log in logging_outputs)
            metrics.log_custom(MicroF1Meter, 'micro_f1', fn, tp, fp, split, weight)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
