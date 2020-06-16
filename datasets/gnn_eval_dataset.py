import logging
from collections import Counter
import math
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from intervaltree import IntervalTree


from fairseq.data import FairseqDataset

from datasets import GraphDataset
from datasets.subgraph_sampler import SubgraphSampler
from utils.data_utils import numpy_seed


logger = logging.getLogger(__name__)


class GNNEvalDataset(FairseqDataset):

    def __init__(
        self,
        annotated_text,
        graph,
        dictionary,
        max_positions,
        num_workers,
        seed,
    ):
        self.annotated_text = annotated_text
        self.graph = graph
        self.dictionary = dictionary
        self.max_positions = max_positions
        self.seed = seed
        self.epoch = None
        self.num_workers = num_workers

    def set_epoch(self, epoch):
        self.graph.set_epoch(epoch)
        self.epoch = epoch

    def sample_relation_statement(self, head, tail, local_interval):
        edges = self.graph.edges[head].numpy().reshape(-1, GraphDataset.EDGE_SIZE)
        left = np.searchsorted(edges[:, GraphDataset.TAIL_ENTITY], tail, side='left')
        right = np.searchsorted(edges[:, GraphDataset.TAIL_ENTITY], tail, side='right')
        indices = np.arange(left, right)
        random_indices = np.random.permutation(indices)
        for idx in random_indices:
            edge = edges[idx]
            # return edge, local_interval
            start = edge[GraphDataset.START_BLOCK]
            end = edge[GraphDataset.END_BLOCK]

            if len(local_interval.overlap(start, end)) == 0:
                local_interval.addi(start, end)
                return edge, local_interval

        return None, local_interval

    def __getitem__(self, index):
        with numpy_seed('GNNEvalDataset', self.seed, self.epoch, index):
            local_interval = IntervalTree()
            edge = self.graph[index]
            head = edge[GraphDataset.HEAD_ENTITY]
            tail = edge[GraphDataset.TAIL_ENTITY]

            start = edge[GraphDataset.START_BLOCK]
            end = edge[GraphDataset.END_BLOCK]
            local_interval.addi(start, end)
            head_neighbors = self.graph.get_neighbors(head)
            tail_neighbors = self.graph.get_neighbors(tail)

            mutual_neighbors = np.intersect1d(head_neighbors, tail_neighbors, assume_unique=True)
            if len(mutual_neighbors) == 0:
                return None

            found_supporting = False
            random_mutual = np.random.permutation(mutual_neighbors)

            for chosen_mutual in random_mutual:
                support1, local_interval = self.sample_relation_statement(head, chosen_mutual, local_interval)
                support2, local_interval = self.sample_relation_statement(chosen_mutual, tail, local_interval)

                if support1 is None or support2 is None:
                    continue
                else:
                    found_supporting = True
                    break

            if found_supporting is False:
                return None

        item = {
            'target': self.annotated_text.annotate(*(edge.numpy())),
            'support': [self.annotated_text.annotate(*(support1)), self.annotated_text.annotate(*(support2))],
            'entities': {'A': head, 'B': tail, 'C': chosen_mutual}
        }

        return item


    def __len__(self):
        return len(self.graph)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return 3 * self.max_positions

    @property
    def sizes(self):
        return np.full(len(self), self.max_positions * 3, dtype=np.int32)

    def ordered_indices(self):
        return self.graph.ordered_indices()

    def collater(self, samples):
        if len(samples) == 0:
            return None

        target_idx = []
        graph_idx = []
        relation_statements = []
        graph_sizes = []
        entities = {'A': [], 'B': [], 'C': []}
        for sample in samples:
            if sample == None:
                continue
            target = sample['target']
            target_idx.append(len(relation_statements))
            relation_statements.append(target)

            supports = sample['support']
            local_support_idx = []
            for support in supports:
                local_support_idx.append(len(relation_statements))
                relation_statements.append(support)

            graph_idx.append(local_support_idx)
            graph_sizes.append(1)

            for entity in entities:
                entities[entity].append(sample['entities'][entity])
        fake_relation_statement = torch.zeros(self.max_positions, dtype=torch.int64)
        relation_statements.append(fake_relation_statement)
        relation_statements = pad_sequence(
            relation_statements,
            batch_first=True,
            padding_value=self.dictionary.pad(),
            )
        relation_statements = relation_statements[:-1]
        # stitch together
        batch = {
            'text': [torch.LongTensor(relation_statements)],
            'graph': torch.LongTensor(graph_idx),
            'graph_sizes': torch.LongTensor(graph_sizes),
            'target_text_idx': torch.LongTensor(target_idx),
            'entities': {key: torch.LongTensor(value) for key, value in entities.items()}
        }

        return batch
