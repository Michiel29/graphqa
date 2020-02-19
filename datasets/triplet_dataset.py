import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch

from datasets import RelInfDataset

class TripletDataset(RelInfDataset):

    def __init__(self, text_data, annotation_data, k_negative, n_entities, dictionary, shift_annotations):
        super().__init__(text_data, annotation_data, k_negative, n_entities, dictionary, shift_annotations)

    def collater(self, instances):
        if len(instances) == 0:
            return None

        mentions = []
        heads = []
        tails = []
        targets = []
        ntokens, nsentences = 0, 0

        for instance in instances:
            mentions.append(instance['mention'])
            heads.extend(instance['head'])
            tails.extend(instance['tail'])
            targets.extend(instance['target'])
            ntokens += instance['ntokens']
            nsentences += instance['nsentences']

        padded_mention = pad_sequence(mentions, batch_first=True, padding_value=self.dictionary.pad())

        return {
            'mention': padded_mention,
            'head':  torch.LongTensor(heads),
            'tail': torch.LongTensor(tails),
            'target':  torch.LongTensor(targets),
            'size': len(instances),
            'ntokens': ntokens,
            'nsentences': nsentences,
        }
