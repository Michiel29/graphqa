import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from datasets import RelInfDataset

class TripletDataset(RelInfDataset):

    def __init__(
        self,
        text_data,
        annotation_data,
        graph,
        k_negative,
        n_entities,
        dictionary,
        shift_annotations,
        mask_type,
    ):
        super().__init__(text_data, annotation_data, graph, k_negative, n_entities, dictionary, shift_annotations, mask_type)

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
            heads.append(instance['head'])
            tails.append(instance['tail'])
            targets.append(instance['target'])
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
