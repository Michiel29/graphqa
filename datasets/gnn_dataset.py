import logging
import math
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import FairseqDataset

from datasets import AnnotatedText, GraphDataset
from datasets.subgraph_sampler import SubgraphSampler
from utils.data_utils import numpy_seed


logger = logging.getLogger(__name__)


class GNNDataset(FairseqDataset):

    def __init__(
        self,
        annotated_text,
        graph,
        dictionary,
        min_common_neighbors,
        max_common_neighbors,
        min_common_neighbors_for_the_last_edge,
        max_entities_size,
        max_entities_from_queue,
        max_tokens,
        max_sentences,
        num_text_chunks,
        seed,
    ):
        self.annotated_text = annotated_text
        self.graph = graph
        self.dictionary = dictionary
        self.min_common_neighbors = min_common_neighbors
        self.max_common_neighbors = max_common_neighbors
        self.min_common_neighbors_for_the_last_edge = min_common_neighbors_for_the_last_edge
        self.max_entities_size = max_entities_size
        self.max_entities_from_queue = max_entities_from_queue
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.num_text_chunks = 4
        self.seed = seed
        self.epoch = None

    def set_epoch(self, epoch):
        self.graph.set_epoch(epoch)
        self.epoch = epoch

    def _sample_subgraph(self, index):
        edge = self.graph[index]
        head = edge[GraphDataset.HEAD_ENTITY]
        tail = edge[GraphDataset.TAIL_ENTITY]

        subgraph = SubgraphSampler(
            graph=self.graph,
            annotated_text=self.annotated_text,
            min_common_neighbors=self.min_common_neighbors,
            max_entities_size=self.max_entities_size,
            max_entities_from_queue=self.max_entities_from_queue,
        )
        sentence = self.annotated_text.annotate(*(edge.numpy()))

        if not subgraph.add_initial_entity_pair(head, tail, self.max_tokens, self.max_sentences, sentence):
            return None

        subgraph.fill(self.max_tokens, self.max_sentences, self.min_common_neighbors_for_the_last_edge)

        return subgraph

    def _split(self, sentences):
        chunk_size = math.ceil(len(sentences) / self.num_text_chunks)
        chunks = []
        for start_pos in range(0, len(sentences), chunk_size):
            end_pos = min(len(sentences), start_pos + chunk_size)
            current_chunk = pad_sequence(
                sentences[start_pos:end_pos],
                batch_first=True,
                padding_value=self.dictionary.pad(),
            )
            chunks.append(current_chunk)
        return chunks

    def _get_all_sentences_and_index(self, subgraph):
        len_and_edges = [(len(sentence), a, b) for (a, b), sentence in subgraph.get_relation_statements().items()]
        len_and_edges.sort()

        sentences = []
        index = {}
        for i, (_, a, b) in enumerate(len_and_edges):
            a_b = (a, b)
            assert a_b not in index
            index[a_b] = i
            sentences.append(subgraph.get_relation_statements()[a_b])

        sentences = self._split(sentences)
        return sentences, index

    def _get_edge_tuples(self, subgraph, index):
        graph = []
        target_text_idx = []
        for a_b in subgraph.get_covered_edges():
            a_b_index = index[a_b]
            current_target_graph = []
            a, b = a_b
            a_b_set = set(a_b)
            for a_c in subgraph.get_relation_statements().keys():
                a_c_set_inter = set(a_c).intersection(a_b_set)
                if not(len(a_c_set_inter) == 1 and next(iter(a_c_set_inter)) == a):
                    continue
                for b_c in subgraph.get_relation_statements().keys():
                    b_c_set_inter = set(b_c).intersection(a_b_set)
                    if not(len(b_c_set_inter) == 1 and next(iter(b_c_set_inter)) == b):
                        continue
                    current_target_graph.append((index[a_c], index[b_c]))
            assert len(current_target_graph) > 0
            if len(current_target_graph) > self.max_common_neighbors:
                current_target_graph = [
                    current_target_graph[x]
                    for x in np.random.permutation(len(current_target_graph))[:self.max_common_neighbors]
                ]
            graph.append(torch.LongTensor(current_target_graph))
            target_text_idx.append(index[a_b])
        target_text_idx = torch.LongTensor(target_text_idx)
        return graph, target_text_idx

    def __getitem__(self, index):
        with numpy_seed('GNNDataset', self.seed, self.epoch, index):
            subgraph = self._sample_subgraph(index)
            while subgraph is None:
                # logging.warning('Failed to sample subgraph for [seed=%d, epoch=%d, index=%d]' % (
                #     self.seed,
                #     self.epoch,
                #     index,
                # ))
                index = np.random.randint(len(self.graph))
                subgraph = self._sample_subgraph(index)

        sentences, index = self._get_all_sentences_and_index(subgraph)
        graph, target_text_idx = self._get_edge_tuples(subgraph, index)

        return {
            'text': sentences,
            'graph': graph,
            'target_text_idx': target_text_idx,
            'target': torch.arange(len(target_text_idx)),
            'yield': subgraph.get_yield(),
            'rel_cov': subgraph.get_relative_coverages_mean(),
            'nsentences': subgraph.nsentences,
            'ntokens': subgraph.ntokens,
        }

    def __len__(self):
        return len(self.graph)

    @property
    def sizes(self):
        return np.full(len(self), self.max_tokens, dtype=np.int32)

    def ordered_indices(self):
        return self.graph.ordered_indices()

    def collater(self, samples):
        if len(samples) == 0:
            return None
        assert len(samples) == 1
        return samples[0]