import numpy as np
import torch
import torch.nn.functional as F

from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion

from utils.diagnostic_utils import Diagnostic
from criterions.cross_entropy import CrossEntropy


@register_criterion('bor')
class BoRCriterion(CrossEntropy):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.task = task

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        model_output = model(sample)
        target = sample['target']

        candidate_similarity = torch.ones_like(model_output) # TODO: replace this placeholder with real candidate_similarity tensor (i.e., sample['candidate_similarity'])
        # candidate_similarity = sample['candidate_similarity']
        model_output = model_output * candidate_similarity

        # diag = Diagnostic(self.task.dictionary, self.task.entity_dictionary, self.task)
        # diag.inspect_batch(sample, scores=model_output)

        loss = F.cross_entropy(model_output, target, reduction='sum' if reduce else 'none')
        pred = torch.argmax(model_output, dim=1)

        sample_size = target.numel()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'sample_size': sample_size,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'accuracy': utils.item((pred == target).float().sum()),
            'num_updates': 1,
        }

        if 'ntokens_AB' in sample.keys():
            logging_output['ntokens_AB'] = sample['ntokens_AB']
        if 'ntokens_mem' in sample.keys():
            logging_output['ntokens_mem'] = sample['ntokens_mem']
        if 'bad_weak_negs' in sample.keys():
            logging_output['bad_weak_negs'] = sample['bad_weak_negs']
        if 'yield' in sample.keys():
            keys = ['yield', 'rel_cov', 'n_mutual_neg', 'n_single_neg', 'n_weak_neg', 'n_mutual_neighbors', 'target_degree']
            for key in keys:
                logging_output[key] = sample[key]

        logging_output = self.task.reporter(target, pred, logging_output)

        return loss, sample_size, logging_output