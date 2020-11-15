import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from fairseq.data import BaseWrapperDataset, data_utils

from utils.data_utils import numpy_seed
from datasets import GraphDataset


logger = logging.getLogger(__name__)


class ETPRelationDownstreamDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        edges,
        dictionary,
        n_entities,
        seed,
        split):

        self.dataset = dataset
        self.edges = edges
        self.dictionary = dictionary
        self.n_entities = n_entities
        self.seed = seed
        self.split = split

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    @property
    def sizes(self):
        return self.dataset.sizes

    def __getitem__(self, index):
        item = self.dataset.__getitem__(index)
        item['target'] = item['answer']
        text = item['question']
        mask_token = torch.LongTensor([self.dictionary.blank()])
        text = torch.cat((text, mask_token), dim=0)
        item['text'] = text
        mask_position = len(text) - 1
        item['mask_annotation'] = torch.LongTensor([(mask_position, mask_position)])
        item['all_annotations'] = torch.LongTensor([entity['position'] for entity in item['annotation']] + [(mask_position, mask_position)])
        item['entity_ids'] = torch.LongTensor([entity.get('entity_id', -1) for entity in item['annotation']] + [-1])

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

        text, mask_annotation, all_annotations, n_annotations, entity_ids, target = [], [], [], [], [], []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            mask_annotation.append(instance['mask_annotation'])
            all_annotations.append(instance['all_annotations'])
            n_annotations.append(instance['all_annotations'].shape[0])
            entity_ids.append(instance['entity_ids'])
            target.append(instance['target'])
            ntokens += len(instance['text'])
            nsentences += 1

        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())

        mask_annotation = torch.cat(mask_annotation)
        all_annotations = torch.cat(all_annotations)
        n_annotations = torch.LongTensor(n_annotations)

        entity_ids = torch.cat(entity_ids, dim=0)

        relation_entity_indices_left = torch.arange(len(entity_ids)).repeat_interleave(n_annotations.repeat_interleave(n_annotations))
        relation_entity_indices_right = [torch.arange(n_annotations[0]).repeat(n_annotations[0])]

        cumulative_idx = n_annotations.cumsum(0)

        for i in range(len(n_annotations) - 1):
            # sample_indices =
            relation_entity_indices_right.append(torch.arange(cumulative_idx[i], cumulative_idx[i] + n_annotations[i + 1]).repeat(n_annotations[i + 1]))
        relation_entity_indices_right = torch.cat(relation_entity_indices_right)


        mask_relation_indices = []
        sample_offset = (n_annotations**2).cumsum(0)
        for i in range(len(n_annotations)):
            mask_relation_indices.append(torch.arange(start=n_annotations[i]-1, end=n_annotations[i]**2 - 1, step=n_annotations[i]) + sample_offset[i] - n_annotations[i]**2)
            mask_relation_indices.append(torch.arange(n_annotations[i] - 1) + sample_offset[i] - n_annotations[i])


        mask_relation_indices = torch.cat(mask_relation_indices)
        mask_throw_away_indices = (n_annotations**2) - 1

        batch = {
            'text': padded_text,
            'target': torch.LongTensor(target),
            'mask_annotation': mask_annotation,
            'all_annotations': all_annotations,
            'n_annotations': n_annotations,
            'relation_entity_indices_left': relation_entity_indices_left,
            'relation_entity_indices_right': relation_entity_indices_right,
            'mask_relation_indices': mask_relation_indices,
            'mask_throw_away_indices': mask_throw_away_indices,
            'entity_ids': entity_ids,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }


        return batch

