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
        total_negatives,
        max_hard_negatives,
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

        self.total_negatives = total_negatives
        self.max_hard_negatives = max_hard_negatives

        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.num_text_chunks = num_text_chunks
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

    def _get_ordered_edge(self, subgraph, a, b):
        if (a, b) in subgraph.get_relation_statements():
            assert (b, a) not in subgraph.get_relation_statements()
            return a, b
        else:
            assert (b, a) in subgraph.get_relation_statements()
            return b, a

    def _get_edge_tuples(self, subgraph, index):
        graph = []
        target_text_idx = []
        for a, b in subgraph.get_covered_edges():
            current_target_graph = []
            mutual_neighbors = subgraph.get_coverage(a, b).both_edges_in_subgraph
            for c in mutual_neighbors:
                current_target_graph.append((
                    index[self._get_ordered_edge(subgraph, a, c)],
                    index[self._get_ordered_edge(subgraph, c, b)],
                ))
            assert len(current_target_graph) > 0
            if len(current_target_graph) > self.max_common_neighbors:
                current_target_graph = [
                    current_target_graph[x]
                    for x in np.random.permutation(len(current_target_graph))[:self.max_common_neighbors]
                ]
            graph.append(torch.LongTensor(current_target_graph))
            target_text_idx.append(index[(a, b)])
        target_text_idx = torch.LongTensor(target_text_idx)
        return graph, target_text_idx

    def _make_negatives(self, subgraph, index):
        graph = []
        candidate_text_idx = []

        for a, b in subgraph.get_covered_edges():
            coverage = subgraph.get_coverage(a, b)
            mutual_neighbors = coverage.both_edges_in_subgraph
            edge_subgraphs = []
            edge_candidate_idx = []
            total_subgraph = []
            for c in mutual_neighbors:
                total_subgraph.append((
                    index[self._get_ordered_edge(subgraph, a, c)],
                    index[self._get_ordered_edge(subgraph, c, b)],
                ))

            total_subgraph = np.array(total_subgraph)

            if len(mutual_neighbors) < 2:
                continue

            n_mutual = len(mutual_neighbors)

            target_leave_out = np.random.randint(n_mutual)
            target_text_idx = index[(a, b)]

            target_neighbor_indices = [i for i in range(n_mutual) if i != target_leave_out]
            target_subgraph = total_subgraph[target_neighbor_indices]
            graph.append(torch.LongTensor(target_subgraph))
            edge_candidate_idx.append(target_text_idx)

            if n_mutual * 2 > self.max_hard_negatives:
                mutual_indices = np.random.choice(n_mutual, size=self.max_hard_negatives // 2, replace=False)
            else:
                mutual_indices = range(n_mutual)

            for mutual_idx in mutual_indices:
                negative_neighbor_indices = [i for i in range(n_mutual) if i != mutual_idx]
                negative_neighbor_subgraph = total_subgraph[negative_neighbor_indices]

                for negative_text_idx in total_subgraph[mutual_idx]:
                    graph.append(torch.LongTensor(negative_neighbor_subgraph))
                    edge_candidate_idx.append(negative_text_idx)

            # If not enough mutual negatives, try negatives with only single neighbor
            if 2 * n_mutual < self.max_hard_negatives:

                neighbors = list(coverage.single_edge_missing)
                n_single_neighbor_negatives = max(0, min(len(neighbors), self.max_hard_negatives - 2 * n_mutual))

                neighbor_choice = np.random.choice(neighbors, size=n_single_neighbor_negatives, replace=False)

                for neighbor in neighbor_choice:
                    for pair in [(a, neighbor), (b, neighbor), (neighbor, a), (neighbor, b)]:
                        if pair in subgraph.get_relation_statements():
                            graph.append(torch.LongTensor(target_subgraph))
                            edge_candidate_idx.append(index[pair])
                            edge_found = True
                            break
                    assert edge_found

            # Finally, add weak negatives
            n_weak_negatives = max(
                0,
                min(
                    self.total_negatives - len(edge_candidate_idx) + 1,
                    len(index) - len(edge_candidate_idx)
                    )
                )
            weak_neg_options = [pair_text_idx for pair_text_idx in index.values() if pair_text_idx not in edge_candidate_idx]
            weak_neg_choices = np.random.choice(weak_neg_options, size=n_weak_negatives, replace=False)

            for pair_text_idx in weak_neg_choices:
                graph.append(torch.LongTensor(target_subgraph))
                edge_candidate_idx.append(pair_text_idx)

            candidate_text_idx.append(edge_candidate_idx)

        candidate_text_idx = torch.LongTensor(candidate_text_idx)

        return graph, candidate_text_idx

    # @profile
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
        graph, candidate_text_idx = self._make_negatives(subgraph, index)

        return {
            'text': sentences,
            'graph': graph,
            'candidate_text_idx': candidate_text_idx,
            'target': torch.zeros(len(candidate_text_idx), dtype=torch.int64),
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
        if len(samples[0]['graph']) == 0:
            return None
        return samples[0]