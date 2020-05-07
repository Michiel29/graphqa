import logging
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import FairseqDataset

from datasets import AnnotatedText, GraphDataset
from subgraph_sampler import SubgraphSampler
from utils.data_utils import numpy_seed


logger = logging.getLogger(__name__)


class R3LDataset(FairseqDataset):

    def __init__(
        self,
        annotated_text,
        graph,
        min_common_neighbors,
        min_common_neighbors_for_the_last_edge,
        max_tokens,
        max_sentences,
        seed,
    ):
        self.annotated_text = annotated_text
        self.graph = graph
        self.min_common_neighbors = min_common_neighbors
        self.min_common_neighbors_for_the_last_edge = min_common_neighbors_for_the_last_edge
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.seed = seed
        self.epoch = None

    def set_epoch(self, epoch):
        self.graph.set_epoch(epoch)
        self.epoch = epoch

    def _sample_subgraph(graph, index):
        edge = self.graph[index]
        head = edge[GraphDataset.HEAD_ENTITY]
        tail = edge[GraphDataset.TAIL_ENTITY]

        subgraph = SubgraphSampler(
            graph=self.graph,
            annotated_text=self.annotated_text,
            min_common_neighbors=self.min_common_neighbors,
        )
        sentence = self.annotated_text.annotate(*(edge.numpy()))

        if not subgraph.add_initial_entity_pair(head, tail, args.max_tokens, args.max_sentences, sentence):
            return None

        subgraph.fill(self.max_tokens, self.max_sentences, self.min_common_neighbors_for_the_last_edge)

        return subgraph

    def _get_all_sentences_and_index(self, subgraph):
        sentences = []
        index = {}
        for i, (head_tail, sentence) in enumerate(subgraph.get_relation_statements()):
            index[head_tail] = i
            sentences.append(sentence)
        sentences = pad_sequence(sentences, batch_first=True, padding_value=self.dictionary.pad())
        return sentences, index

    def _get_edge_tuples(self, subgraph, index):
        targets = {}
        for a_b in subgraph.get_covered_edges():
            a_b_index = index[a_b]
            targets[a_b_index] = []
            a, b = a_b
            a_b_set = set(a_b)
            for a_c in subgraph.get_relation_statements().keys():
                a_c_set_diff = set(a_c).difference(a_b_set)
                if len(a_c_set_diff) != 1 or next(iter(a_c_set_diff)) != a:
                    continue
                for b_c in subgraph.get_relation_statements().keys():
                    b_c_set_diff = set(b_c).difference(a_b_set)
                    if len(b_c_set_diff) != 1 or next(iter(b_c_set_diff)) != b:
                        continue
                targets[a_b_index].append((index[a_c], index[b_c]))
            assert len(targets[a_b_index]) > 0
        return targets

    def __getitem__(self, index):
        with numpy_seed('R3LDataset', self.seed, self.epoch, index):
            subgraph = _sample_subgraph(index)
            while subgraph is None:
                logging.warning('Failed to sample subgraph for [seed=%d, epoch=%d, index=%d]' % (
                    self.seed,
                    self.epoch,
                    index,
                ))
                index = np.random.randint(len(self.graph))
                subgraph = _sample_subgraph(index)

        sentences, index = self._get_all_sentences_and_index(subgraph)
        graph = self._get_edge_tuples(subgraph, index)

        return {
            'text': sentences,
            'graph': graph,
            'yield': subgraph.get_yield(),
            'rel_cov': subgraph.get_relative_coverages_mean(),
            'nsentences': subgraph.nsentences,
            'ntokens': subgraph.ntokens,
        }

    def __len__(self):
        return len(self.graph)

    def ordered_indices(self):
        return self.graph.ordered_indices()

    def collater(self, samples):
        assert len(samples) == 1
        return samples[0]