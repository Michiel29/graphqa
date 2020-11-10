import logging
import math
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import FairseqDataset

from datasets import GraphDataset
from utils.data_utils import numpy_seed


logger = logging.getLogger(__name__)


class ETPRelationDataset(FairseqDataset):

    def __init__(
        self,
        annotated_text,
        edges,
        dictionary,
        n_entities,
        total_negatives,
        mask_negative_prob,
        max_positions,
        num_workers,
        seed,
    ):
        self.annotated_text = annotated_text
        self.edges = edges
        self.dictionary = dictionary
        # Could take len(edges) instead of passing number of entities, but old indexed dataset had extra Null entities
        self.n_entities = n_entities
        self.total_negatives = total_negatives
        self.mask_negative_prob = mask_negative_prob

        self.seed = seed
        self.epoch = None
        self._sizes = np.full(len(self), max_positions, dtype=np.int64)

        self.num_workers = num_workers

    def set_epoch(self, epoch):
        self.epoch = epoch


    def __getitem__(self, index):
        with numpy_seed('ETPRelationDataset', self.seed, self.epoch, index):
            sampled_edge = None
            while not sampled_edge:
                entity = np.random.randint(len(self.edges))
                n_entity_edges = len(self.edges[entity]) // GraphDataset.EDGE_SIZE
                if n_entity_edges > 0:
                    passage_idx = np.random.randint(n_entity_edges)
                    edge_start = passage_idx * GraphDataset.EDGE_SIZE
                    edge = self.edges[entity][edge_start:edge_start + GraphDataset.EDGE_SIZE].numpy()
                    sampled_edge = True

            start_pos, end_pos, start_block, end_block = edge[GraphDataset.HEAD_START_POS], edge[GraphDataset.HEAD_END_POS], edge[GraphDataset.START_BLOCK], edge[GraphDataset.END_BLOCK]
            passage, mask_annotation_position, all_annotation_positions, entity_ids = self.annotated_text.annotate_mention(entity, start_pos, end_pos, start_block, end_block, return_all_annotations=True)

            mask_idx = all_annotation_positions.index(mask_annotation_position[0])

            replace_probs = np.ones(len(all_annotation_positions)) * (1-self.mask_negative_prob) / (len(all_annotation_positions) - 1)
            replace_probs[mask_idx] = self.mask_negative_prob
            replace_position = np.random.choice(len(all_annotation_positions), size=1, p=replace_probs)

            entity_replacements = np.random.choice(self.n_entities, replace=False, size=self.total_negatives)

            # candidates = np.expand_dims(entity_ids, axis=-1).repeat(self.total_negatives, axis=-1)
            # candidates[replace_position, np.arange(len(replace_position))] = entity_replacements

        item = {
            'text': passage,
            'mask_annotation': torch.LongTensor(mask_annotation_position),
            'all_annotations': torch.LongTensor(all_annotation_positions),
            'entity_ids': torch.LongTensor(entity_ids),
            'entity_replacements': torch.LongTensor(entity_replacements),
            'replacement_position': replace_position,
            # 'candidates': torch.LongTensor(candidates),
        }
        return item

    def __len__(self):
        return len(self.edges)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @property
    def sizes(self):
        return self._sizes

    def ordered_indices(self):
        return np.random.permutation(len(self))

    def collater(self, instances):
        batch_size = len(instances)

        if batch_size == 0:
            return None

        text, mask_annotation, all_annotations, n_annotations, entity_ids, entity_replacements, replacement_positions = [], [], [], [], [], [], []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            mask_annotation.append(instance['mask_annotation'])
            all_annotations.append(instance['all_annotations'])
            n_annotations.append(instance['all_annotations'].shape[0])
            entity_ids.append(instance['entity_ids'])
            entity_replacements.append(instance['entity_replacements'])
            replacement_positions.append(instance['replacement_position'])
            ntokens += len(instance['text'])
            nsentences += 1

        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())

        mask_annotation = torch.cat(mask_annotation)
        all_annotations = torch.cat(all_annotations)
        n_annotations = torch.LongTensor(n_annotations)

        entity_ids = torch.cat(entity_ids, dim=0)
        entity_replacements = torch.stack(entity_replacements, dim=0)
        replacement_positions = torch.LongTensor(replacement_positions).squeeze()
        # replacement_positions[1:] = replacement_positions[1:] + n_annotations.cumsum(0)[:-1]

        relation_entity_indices_left = torch.arange(len(entity_ids)).repeat_interleave(n_annotations.repeat_interleave(n_annotations))
        relation_entity_indices_right = [torch.arange(n_annotations[0]).repeat(n_annotations[0])]

        cumulative_idx = n_annotations.cumsum(0)
        for i in range(len(n_annotations) - 1):
            # sample_indices =
            relation_entity_indices_right.append(torch.arange(cumulative_idx[i], cumulative_idx[i] + n_annotations[i + 1]).repeat(n_annotations[i + 1]))
        relation_entity_indices_right = torch.cat(relation_entity_indices_right)

        replacement_relation_indices_left, replacement_relation_indices_right = [], []

        offset = 0
        for i in range(len(n_annotations)):

            sample_replacement_indices_left = offset + replacement_positions[i] * n_annotations[i] + torch.arange(n_annotations[i])
            sample_replacement_indices_right = offset + replacement_positions[i] + torch.arange(0, n_annotations[i]**2, n_annotations[i])
            replacement_relation_indices_left.append(sample_replacement_indices_left)
            replacement_relation_indices_right.append(sample_replacement_indices_right)
            offset += n_annotations[i]**2

        replacement_relation_indices_left = torch.cat(replacement_relation_indices_left, dim=0)
        replacement_relation_indices_right = torch.cat(replacement_relation_indices_right, dim=0)
        replacement_self_indices = torch.LongTensor([replacement_positions[0]] + [replacement_positions[i + 1] + cumulative_idx[i] for i in range(len(cumulative_idx) - 1)])




        batch = {
            'text': padded_text,
            'target': torch.zeros(batch_size, dtype=torch.int64),
            'mask_annotation': mask_annotation,
            'all_annotations': all_annotations,
            'n_annotations': n_annotations,
            'relation_entity_indices_left': relation_entity_indices_left,
            'relation_entity_indices_right': relation_entity_indices_right,
            'entity_ids': entity_ids,
            'entity_replacements': entity_replacements,
            'replacement_positions': replacement_positions,
            'replacement_relation_indices_left': replacement_relation_indices_left,
            'replacement_relation_indices_right': replacement_relation_indices_right,
            'replacement_self_indices': replacement_self_indices,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }

        return batch
