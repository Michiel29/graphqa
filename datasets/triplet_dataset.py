import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from datasets import RelInfDataset


class TripletDataset(RelInfDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def collater(self, instances):
        if len(instances) == 0:
            return None

        texts = []
        heads = []
        tails = []
        targets = []
        ntokens, nsentences = 0, 0

        for instance in instances:
            texts.append(instance['text'])
            heads.append(instance['head'])
            tails.append(instance['tail'])
            targets.append(instance['target'])
            ntokens += instance['ntokens']
            nsentences += instance['nsentences']

        padded_text = pad_sequence(texts, batch_first=True, padding_value=self.dictionary.pad())

        return {
            'text': padded_text,
            'head':  torch.LongTensor(heads),
            'tail': torch.LongTensor(tails),
            'target':  torch.LongTensor(targets),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': nsentences
        }
