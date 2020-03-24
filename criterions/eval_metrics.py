import numpy as np
from sklearn.metrics import f1_score

class Metric():
    def reset_metrics(self):
        return NotImplementedError

    def update_metrics(self, pred, target, batch_size, data):
        return NotImplementedError

class F1Score(Metric):
    def __init__(self):
        self.f1_value = None

    def reset_metrics(self):        
        self.results_dict = {'pred': None, 'target': None}

    def update_metrics(self, pred, target):
        if self.results_dict['pred'] is None:
            self.results_dict['pred'] = pred.detach().cpu().numpy()
        else:
            self.results_dict['pred'] = np.concatenate((self.results_dict['pred'], pred.detach().cpu().numpy()), axis=0)

        if self.results_dict['target'] is None:
            self.results_dict['target'] = target.cpu().numpy()
        else:
            self.results_dict['target'] = np.concatenate((self.results_dict['target'], target.cpu().numpy()), axis=0)

        self.f1_value = self.metric()

    def metric(self):
        pred = self.results_dict['pred']
        target = self.results_dict['target']
        f1 = f1_score(target, pred, average='micro')
        return f1