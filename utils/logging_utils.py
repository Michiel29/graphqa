import numpy as np
import collections
from fairseq.logging.meters import AverageMeter, safe_round
from typing import Optional
# TODO: Uncomment if the class is used
from sklearn.metrics import multilabel_confusion_matrix


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

    def update(self, fn, tp, fp, task, split, avg, n=1):
        for i in fn.keys():
            self.fn[i] += fn[i]
            self.tp[i] += tp[i]
            self.fp[i] += fp[i]
        self.split = split
        self.count += n
        self.batch_f1 = compute_macro_f1(fn, tp, fp, task)
        self.batch_f1_sum += self.batch_f1 * n
        self.epoch_f1 = compute_macro_f1(self.fn, self.tp, self.fp, task)

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

def compute_macro_f1(fn, tp, fp, task=None):
    f1_sum = 0
    if task in ['kbp37', 'semeval2010task8']:
        rel_indices = range(0, len(fn)-1, 2)
        for i in rel_indices:
            f1_sum += compute_f1(fn[i]+fn[i+1], tp[i]+tp[i+1], fp[i]+fp[i+1])
        macro_f1 = f1_sum / len(rel_indices)
    else:
        for i in fn.keys():
            f1_sum += compute_f1(fn[i], tp[i], fp[i])
        macro_f1 = f1_sum / len(fn)
    return macro_f1

