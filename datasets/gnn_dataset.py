import logging
from collections import Counter
import math
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import FairseqDataset

from datasets import GraphDataset
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
        required_min_common_neighbors,
        max_entities_size,
        max_entities_from_queue,
        cover_random_prob,
        total_negatives,
        max_hard_negatives,
        max_tokens,
        max_sentences,
        num_text_chunks,
        entity_pair_counter_cap,
        num_workers,
        seed,
    ):
        self.annotated_text = annotated_text
        self.graph = graph
        self.dictionary = dictionary

        self.min_common_neighbors = min_common_neighbors
        self.max_common_neighbors = max_common_neighbors
        self.required_min_common_neighbors = required_min_common_neighbors
        self.max_entities_size = max_entities_size
        self.max_entities_from_queue = max_entities_from_queue
        self.cover_random_prob = cover_random_prob

        self.total_negatives = total_negatives
        self.max_hard_negatives = max_hard_negatives

        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.num_text_chunks = num_text_chunks
        self.seed = seed
        self.epoch = None

        self.num_workers = num_workers
        self.entity_pair_counter_cap = entity_pair_counter_cap
        if self.entity_pair_counter_cap is not None:
            self.entity_pair_counter = Counter()
            self.entity_pair_counter_sum = 0
        else:
            self.entity_pair_counter = None
            self.entity_pair_counter_sum = None

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
            required_min_common_neighbors=self.required_min_common_neighbors,
            max_entities_size=self.max_entities_size,
            max_entities_from_queue=self.max_entities_from_queue,
            cover_random_prob=self.cover_random_prob,
            entity_pair_counter=self.entity_pair_counter,
            entity_pair_counter_sum=self.entity_pair_counter_sum,
            entity_pair_counter_cap=self.entity_pair_counter_cap,
        )
        sentence = self.annotated_text.annotate_relation(*(edge.numpy()))

        if not subgraph.add_initial_entity_pair(head, tail, self.max_tokens, self.max_sentences, sentence):
            return None

        subgraph.fill(self.max_tokens, self.max_sentences)
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

    def _make_negatives(self, subgraph, index):
        graph = []
        graph_sizes = []
        candidate_text_idx = []

        target_degree = 0
        num_mutual_neighbors = 0
        num_mutual_negatives, num_single_negatives, num_weak_negatives = 0, 0, 0

        for a, b in subgraph.get_covered_edges():
            coverage = subgraph.coverage[(a, b)]
            mutual_neighbors = coverage.both_edges_in_subgraph

            n_mutual = len(mutual_neighbors)
            num_mutual_neighbors += n_mutual
            target_degree += (self.graph.get_degree(a) + self.graph.get_degree(b))

            edge_subgraphs = []
            edge_candidate_idx = []
            total_subgraph = []
            for c in mutual_neighbors:
                total_subgraph.append((index[(a, c)], index[(c, b)]))

            total_subgraph = np.array(total_subgraph)

            if n_mutual > self.max_common_neighbors:
                target_neighbor_indices = np.random.choice(n_mutual, size=self.max_common_neighbors, replace=False)
                target_subgraph = total_subgraph[target_neighbor_indices]
                edge_subgraphs.append(target_subgraph)
                edge_candidate_idx.append(index[(a, b)])

                possible_hard_negatives = np.array([idx for idx in range(n_mutual) if idx not in target_neighbor_indices])

                if len(possible_hard_negatives) * 2 > self.max_hard_negatives:
                    hard_negatives = np.random.choice(possible_hard_negatives, size=self.max_hard_negatives // 2, replace=False)
                else:
                    hard_negatives = possible_hard_negatives

                for hard_negative in hard_negatives:
                    for negative_text_idx in total_subgraph[hard_negative]:
                        edge_subgraphs.append(target_subgraph)
                        edge_candidate_idx.append(negative_text_idx)
                        num_mutual_negatives += 1
            else:
                target_subgraph = total_subgraph
                edge_subgraphs.append(target_subgraph)
                edge_candidate_idx.append(index[(a, b)])

            # If not enough mutual negatives, try negatives with only single neighbor
            if len(edge_candidate_idx) < self.max_hard_negatives:

                neighbors = list(coverage.single_edge_missing)
                n_single_neighbor_negatives = max(0, min(len(neighbors), self.max_hard_negatives - 2 * n_mutual))

                neighbor_choice = np.random.choice(neighbors, size=n_single_neighbor_negatives, replace=False)

                for neighbor in neighbor_choice:
                    for pair in [(a, neighbor), (b, neighbor), (neighbor, a), (neighbor, b)]:
                        if pair in subgraph.get_relation_statements():
                            edge_subgraphs.append(target_subgraph)
                            edge_candidate_idx.append(index[pair])
                            num_single_negatives += 1
                            edge_found = True
                            break
                    assert edge_found

            # Finally, add weak negatives
            n_weak_negatives = max(
                0,
                min(
                    self.total_negatives - len(edge_candidate_idx) + 1,
                    len(index) - len(edge_candidate_idx),
                ),
            )
            weak_neg_options = [pair_text_idx for pair_text_idx in index.values() if pair_text_idx not in edge_candidate_idx]
            weak_neg_choices = np.random.choice(weak_neg_options, size=n_weak_negatives, replace=False)

            for pair_text_idx in weak_neg_choices:
                edge_subgraphs.append(target_subgraph)
                edge_candidate_idx.append(pair_text_idx)
                num_weak_negatives += 1

            graph_sizes.extend([len(g) for g in edge_subgraphs])
            graph.append(torch.LongTensor(edge_subgraphs).reshape(-1, 2))
            candidate_text_idx.append(edge_candidate_idx)


        graph = torch.cat(graph, dim=0)
        graph_sizes = torch.LongTensor(graph_sizes)
        candidate_text_idx = torch.LongTensor(candidate_text_idx)
        num_positive_examples = float(len(candidate_text_idx))

        logging_output = {
            'target_degree': target_degree / num_positive_examples / 2,
            'n_mutual_neg': num_mutual_negatives / num_positive_examples,
            'n_single_neg': num_single_negatives / num_positive_examples,
            'n_weak_neg': num_weak_negatives / num_positive_examples,
            'n_mutual_neighbors': num_mutual_neighbors / num_positive_examples,
        }

        return graph, graph_sizes, candidate_text_idx, logging_output

    def __getitem__(self, index):
        with numpy_seed('GNNDataset', self.seed, self.epoch, index):
            subgraph = self._sample_subgraph(index)
            while subgraph is None:
                # logging.warning('Failed to sample subgraph for [seed=%d, epoch=%d, index=%d]' % (
                #     self.seed,
                #     self.epoch,
                #     index,
                # ))
                text_index = np.random.randint(len(self.graph))
                subgraph = self._sample_subgraph(text_index)

        sentences, text_index = self._get_all_sentences_and_index(subgraph)
        graph, graph_sizes, candidate_text_idx, logging_output = self._make_negatives(subgraph, text_index)

        item = {
            'text': sentences,
            'graph': graph,
            'graph_sizes': graph_sizes,
            'candidate_text_idx': candidate_text_idx,
            'target': torch.zeros(len(candidate_text_idx), dtype=torch.int64),
            'all_entity_pairs': torch.tensor(list(text_index.keys()), dtype=torch.int32),
            'yield': subgraph.get_yield(),
            'rel_cov': subgraph.get_relative_coverages_mean(),
            'nsentences': subgraph.nsentences,
            'ntokens': subgraph.ntokens,
        }
        item.update(logging_output)
        return item

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

        if self.entity_pair_counter_cap is not None and self.num_workers > 0:
            self.entity_pair_counter.update([
                (entity_pair[0].item(), entity_pair[1].item())
                for entity_pair in samples[0]['all_entity_pairs']
            ])
            self.entity_pair_counter_sum += len(samples[0]['all_entity_pairs'])

        return samples[0]
