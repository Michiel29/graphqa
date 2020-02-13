import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch

from datasets import RelInfDataset

class TripletDataset(RelInfDataset):

    def __init__(self, text_data, annotation_data, k_negative, n_entities, dictionary, n_examples):
        super().__init__(text_data, annotation_data, k_negative, n_entities, dictionary, n_examples)

    def collater(self, instances):

        mentions = []
        heads = []
        tails = []

        for instance in instances:

            """Perform Masking"""
            mention, instance_heads, instance_tails = self.sample_entities(instance)
            mentions.append(mention)
            heads.append(instance_heads)
            tails.append(instance_tails)

        padded_mention = pad_sequence(mentions, batch_first=True, padding_value=self.dictionary.pad())

        batch = {}
        batch['mention'] = padded_mention
        batch['head'] =  torch.LongTensor(heads)
        batch['tail'] = torch.LongTensor(tails)
        batch['target'] = torch.zeros(len(instances), dtype=torch.long)
        batch['batch_size'] = len(instances)
        batch['ntokens'] = sum(len(m) for m in mentions)
        batch['nsentences'] = len(instances)

        return batch
