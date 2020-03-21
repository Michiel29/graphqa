import numpy.random as rd
import torch

from fairseq.data import FairseqDataset
from fairseq.data import data_utils

from datasets import AnnotatedText, GraphDataset


class RelInfDataset(FairseqDataset):

    def __init__(
        self,
        annotated_text,
        graph,
        k_negative,
        n_entities,
        seed,
    ):
        self.annotated_text = annotated_text
        self.k_negative = k_negative
        self.n_entities = n_entities
        self.graph = graph
        self.seed = seed
        self.epoch = None

    def set_epoch(self, epoch):
        self.graph.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index):
        edge = self.graph[index]
        head = edge[GraphDataset.HEAD_ENTITY]
        tail = edge[GraphDataset.TAIL_ENTITY]
        item = {}

        with data_utils.numpy_seed(17101990, self.seed, self.epoch, index):
            item['text'] = self.annotated_text.annotate(*(edge.numpy()))
            item['nsentences'] = 1
            item['ntokens'] = len(item['text'])

            replace_heads = rd.randint(2, size=self.k_negative)

            head_neighbors = self.graph.get_neighbors(head)
            tail_neighbors = self.graph.get_neighbors(tail)

            tail_head_neighbors = [tail_neighbors, head_neighbors]

            replacement_entities = []

            for replace_head in replace_heads:
                replacement_neighbors, static_neighbors = tail_head_neighbors[replace_head], tail_head_neighbors[1 - replace_head]

                if len(replacement_neighbors) > 0:
                    replacement_entity = rd.choice(replacement_neighbors)
                elif len(static_neighbors) > 0:
                    replacement_entity = rd.choice(static_neighbors)
                else:
                    replacement_entity = rd.randint(self.n_entities)

                replacement_entities.append(replacement_entity)

            item['head'] = [head] + [head if not replace_heads[i] else replacement_entities[i] for i in range(self.k_negative)]
            item['tail'] = [tail] + [tail if replace_heads[i] else replacement_entities[i] for i in range(self.k_negative)]
            item['target'] = 0

        return item

    def __len__(self):
        return len(self.graph)

    def num_tokens(self, index):
        return self.graph.sizes[index]

    def size(self, index):
        return self.graph.sizes[index]

    @property
    def sizes(self):
        return self.graph.sizes

    def ordered_indices(self):
        return self.graph.ordered_indices()