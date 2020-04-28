import collections
import neptune
from numbers import Number
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, log_loss, accuracy_score
from typing import Optional

from fairseq import meters, distributed_utils
from fairseq.logging.meters import AverageMeter, safe_round
from fairseq.logging.progress_bar import BaseProgressBar


class NeptuneWrapper(BaseProgressBar):
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def __iter__(self):
        return iter(self.progress_bar)

    def log(self, stats, tag=None, step=None):
        self._log_to_neptune(stats, tag=tag, step=step)
        self.progress_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        self._log_to_neptune(stats, tag=tag, step=step)
        self.progress_bar.print(stats, tag=tag, step=step)

    def _get_log_name(self, tag, key):
        if tag is None:
            return key
        else:
            return '%s/%s' % (tag, key)

    def _log_to_neptune(self, stats, tag=None, step=None):
        if step is None:
            step = stats['num_updates']
        for key in stats.keys() - {'num_updates'}:
            if isinstance(stats[key], AverageMeter):
                neptune.log_metric(
                    log_name=self._get_log_name(tag, key),
                    x=step,
                    y=stats[key].val,
                )
            elif isinstance(stats[key], Number):
                neptune.log_metric(
                    log_name=self._get_log_name(tag, key),
                    x=step,
                    y=stats[key],
                )


def maybe_wrap_neptune_logging(progress_bar, args):
    if distributed_utils.is_master(args) and not args.debug:
        return NeptuneWrapper(progress_bar)
    else:
        return progress_bar


class MacroF1Meter(AverageMeter):

    def __init__(self, round: Optional[int] = 3):
        self.round = round
        self.reset()
        self.split = None

    def reset(self):
        self.count = 0
        self.fn = collections.defaultdict(int)
        self.tp = collections.defaultdict(int)
        self.fp = collections.defaultdict(int)
        self.batch_f1 = 0
        self.batch_f1_sum = 0
        self.epoch_f1 = 0

    def update(self, fn, tp, fp, split, ignore_classes=[], aggregate_class_pairs=False, n=1):
        for i in fn.keys():
            self.fn[i] += fn[i]
            self.tp[i] += tp[i]
            self.fp[i] += fp[i]
        self.split = split
        self.count += n
        self.batch_f1 = compute_macro_f1(fn, tp, fp, ignore_classes, aggregate_class_pairs)
        self.batch_f1_sum += self.batch_f1 * n
        self.epoch_f1 = compute_macro_f1(self.fn, self.tp, self.fp, ignore_classes, aggregate_class_pairs)

    def state_dict(self):
        return {
            'fn': self.fn,
            'tp': self.tp,
            'fp': self.fp,
            'split': self.split,
            'count': self.count,
            'round': self.round,
            'batch_f1': self.batch_f1,
            'batch_f1_sum': self.batch_f1_sum,
            'epoch_f1': self.epoch_f1
        }

    def load_state_dict(self, state_dict):
        self.fn = state_dict['fn']
        self.tp = state_dict['tp']
        self.fp = state_dict['fp']
        self.split = state_dict['split']
        self.count = state_dict['count']
        self.round = state_dict.get('round', None)
        self.batch_f1 = state_dict['batch_f1']
        self.batch_f1_sum = state_dict['batch_f1_sum']
        self.epoch_f1 = state_dict['epoch_f1']

    @property
    def avg(self):
        if self.split == 'train':
            return self.batch_f1_sum / self.count if self.count > 0 else self.batch_f1
        elif self.split == 'valid':
            return self.epoch_f1
        else:
            raise NotImplementedError

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val

class MicroF1Meter(AverageMeter):

    def __init__(self, round: Optional[int] = 3):
        self.round = round
        self.reset()
        self.split = None

    def reset(self):
        self.count = 0
        self.fn = 0
        self.tp = 0
        self.fp = 0
        self.batch_f1 = 0
        self.batch_f1_sum = 0
        self.epoch_f1 = 0

    def update(self, fn, tp, fp, split, n=1):
        self.fn += fn
        self.tp += tp
        self.fp += fp
        self.split = split
        self.count += n
        self.batch_f1 = compute_f1(fn, tp, fp)
        self.batch_f1_sum += self.batch_f1 * n
        self.epoch_f1 = compute_f1(self.fn, self.tp, self.fp)

    def state_dict(self):
        return {
            'fn': self.fn,
            'tp': self.tp,
            'fp': self.fp,
            'split': self.split,
            'count': self.count,
            'round': self.round,
            'batch_f1': self.batch_f1,
            'batch_f1_sum': self.batch_f1_sum,
            'epoch_f1': self.epoch_f1
        }

    def load_state_dict(self, state_dict):
        self.fn = state_dict['fn']
        self.tp = state_dict['tp']
        self.fp = state_dict['fp']
        self.split = state_dict['split']
        self.count = state_dict['count']
        self.round = state_dict.get('round', None)
        self.batch_f1 = state_dict['batch_f1']
        self.batch_f1_sum = state_dict['batch_f1_sum']
        self.epoch_f1 = state_dict['epoch_f1']

    @property
    def avg(self):
        if self.split == 'train':
            return self.batch_f1_sum / self.count if self.count > 0 else self.batch_f1
        elif self.split == 'valid':
            return self.epoch_f1
        else:
            raise NotImplementedError

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val

def compute_confusion_matrix(target, pred, avg, num_classes, ignore_classes=[]):
    mcm = multilabel_confusion_matrix(target, pred, labels=list(range(num_classes)))
    if avg == 'macro':
        fn = collections.defaultdict(int)
        tp = collections.defaultdict(int)
        fp = collections.defaultdict(int)
        for i in range(mcm.shape[0]):
            cur_mcm = mcm[i]
            fn[i], tp[i], fp[i] = cur_mcm[1, 0], cur_mcm[1, 1], cur_mcm[0, 1]
    elif avg == 'micro':
        mcm = np.delete(mcm, ignore_classes, axis=0)
        micro_mcm = np.sum(mcm, axis=0)
        fn, tp, fp = micro_mcm[1, 0], micro_mcm[1, 1], micro_mcm[0, 1]
    return fn, tp, fp

def compute_f1(fn, tp, fp):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def compute_macro_f1(fn, tp, fp, ignore_classes=[], aggregate_class_pairs=False):
    f1_sum = 0
    class_indices = np.array(list(fn.keys()))
    class_indices = np.delete(class_indices, ignore_classes)
    if aggregate_class_pairs:
        assert len(class_indices) % 2 == 0
        class_indices = class_indices[::2]
        for i in class_indices:
            f1_sum += compute_f1(fn[i]+fn[i+1], tp[i]+tp[i+1], fp[i]+fp[i+1])
    else:
        for i in class_indices:
            f1_sum += compute_f1(fn[i], tp[i], fp[i])
    macro_f1 = f1_sum / len(class_indices)
    return macro_f1

def reduce_macro_mcm(logging_outputs, num_classes, prefix):
    fn = collections.defaultdict(int)
    tp = collections.defaultdict(int)
    fp = collections.defaultdict(int)
    for i in range(num_classes):
        fn[i] = sum(log.get(prefix + 'fn_' + str(i), 0) for log in logging_outputs)
        tp[i] = sum(log.get(prefix + 'tp_' + str(i), 0) for log in logging_outputs)
        fp[i] = sum(log.get(prefix + 'fp_' + str(i), 0) for log in logging_outputs)
    return fn, tp, fp

def compute_sklearn_stats(target, pred, prob, num_classes, eval_metric):
    stats = {
        'loss': log_loss(target, prob, labels=list(range(num_classes))),
        'acc': accuracy_score(target, pred)
    }

    if eval_metric == 'macro_f1':
        fn, tp, fp = compute_confusion_matrix(
            target=target,
            pred=pred,
            avg='macro',
            num_classes=num_classes
        )
        stats['macro_f1'] = compute_macro_f1(fn, tp, fp, [num_classes-1], True)

    elif eval_metric == 'micro_f1':
        fn, tp, fp = compute_confusion_matrix(
            target=target,
            pred=pred,
            avg='micro',
            num_classes=num_classes,
            ignore_classes=[num_classes-1]
        )
        stats['micro_f1'] = compute_f1(fn, tp, fp)

    return stats

meters.__dict__['MacroF1Meter'] = MacroF1Meter
meters.__dict__['MicroF1Meter'] = MicroF1Meter