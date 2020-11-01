import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import BaseWrapperDataset, data_utils
from utils.data_utils import numpy_seed

logger = logging.getLogger(__name__)


class ETPDownstreamDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        edges,
        dictionary,
        n_entities,
        total_negatives,
        seed,
        split):

        self.dataset = dataset
        self.edges = edges
        self.dictionary = dictionary
        self.n_entities = n_entities
        self.total_negatives = total_negatives
        self.seed = seed
        self.split = split

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    @property
    def sizes(self):
        return self.dataset.sizes

    def __getitem__(self, index, **kwargs):
        item = self.dataset.__getitem__(index, **kwargs)
        answer_entity = item['answer']
        with data_utils.numpy_seed('etp_downstream', self.seed, self.epoch, index):
            negatives = np.random.choice(self.n_entities, replace=False, size=self.total_negatives)
            candidates = np.concatenate(([answer_entity], negatives))
            item['candidates'] = candidates

        if self.split == 'train':
            target = 0
        else:
            target = answer_entity

        item['target'] = target

        text = item['question']
        mask_token = torch.LongTensor([self.dictionary.blank()])
        text = torch.cat((text, mask_token), dim=0)
        item['text'] = text
        mask_position = len(text) - 1
        item['annotation'] = torch.LongTensor([(mask_position, mask_position)])

        return item

    def __len__(self):
        return len(self.dataset)

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size[index]

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    def collater(self, instances):
        batch_size = len(instances)

        if batch_size == 0:
            return None

        text, annotation, candidates, target = [], [], [], []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            annotation.append(instance['annotation'])
            candidates.append(instance['candidates'])
            target.append(instance['target'])
            ntokens += len(instance['text'])
            nsentences += 1

        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())

        if len(annotation) > 0 and annotation[0] is not None:
            annotation = torch.cat(annotation)
        else:
            annotation = None

        batch = {
            'text': padded_text,
            'target': torch.LongTensor(target),
            'annotation': annotation,
            'candidates': torch.LongTensor(candidates),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }

        return batch

