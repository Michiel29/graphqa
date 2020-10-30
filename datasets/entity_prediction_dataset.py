import logging
import math
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import FairseqDataset

from datasets import GraphDataset
from utils.data_utils import numpy_seed


logger = logging.getLogger(__name__)


class EntityPredictionDataset(FairseqDataset):

    def __init__(
        self,
        annotated_text,
        edges,
        dictionary,
        n_entities,
        total_negatives,
        max_positions,
        seed,
    ):
        self.annotated_text = annotated_text
        self.edges = edges
        self.dictionary = dictionary
        # Could take len(edges) instead of passing number of entities, but old indexed dataset had extra Null entities
        self.n_entities = n_entities
        self.total_negatives = total_negatives

        self.seed = seed
        self.epoch = None
        self._sizes = np.full(len(self), max_positions, dtype=np.int64)


    def set_epoch(self, epoch):
        self.epoch = epoch


    def __getitem__(self, index):
        with numpy_seed('EntityPredictionDataset', self.seed, self.epoch, index):
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
            passage, annotation_position = self.annotated_text.annotate_mention(entity, start_pos, end_pos, start_block, end_block)
            negatives = np.random.choice(self.n_entities, replace=False, size=self.total_negatives)
            candidates = np.concatenate(([entity], negatives))

        item = {
            'text': passage,
            'annotation': torch.LongTensor(annotation_position) if annotation_position else None,
            'candidates': candidates,
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

        text, annotation, candidates = [], [], []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            annotation.append(instance['annotation'])
            candidates.append(instance['candidates'])
            ntokens += len(instance['text'])
            nsentences += 1

        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())

        if len(annotation) > 0 and annotation[0] is not None:
            annotation = torch.cat(annotation)
        else:
            annotation = None

        batch = {
            'text': padded_text,
            'target': torch.zeros(batch_size, dtype=torch.int64),
            'annotation': annotation,
            'candidates': torch.LongTensor(candidates),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }

        return batch
