import heapq
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import logging


from datasets import GraphDataset
from datasets.subgraph_sampler_faster import NeighborhoodCoverage, update_coverage


logger = logging.getLogger(__name__)


def item(tensor):
    if isinstance(tensor, torch.Tensor):
        assert tensor.numel() == 1
        return tensor.item()
    else:
        return tensor


class SubgraphSampler(object):

    def __init__(
        self,
        graph,
        annotated_text,
        min_common_neighbors,
        max_entities_size,
        max_entities_from_queue,
    ):
        self.graph = graph
        self.annotated_text = annotated_text
        self.min_common_neighbors = min_common_neighbors
        self.max_entities_size = max_entities_size
        self.max_entities_from_queue = max_entities_from_queue
        self.entities = set([])
        self.entity_pairs = set([])
        self.covered_entity_pairs = set([])
        self.coverage = {}
        self.relation_statements = {}
        self.ntokens = 0
        self.nsentences = 0

        self.entity_score = []
        self.entity_coverage = {}

    def _update_coverage(self, new_relation_statements=None):
        update_coverage(
            self.graph,
            self.entities,
            self.entity_pairs,
            self.coverage,
            self.min_common_neighbors,
            new_relation_statements,
        )

    def get_coverage(self, a, b):
        return self.coverage.get((min(a, b), max(a, b)), None)

    def _add_entities_with_the_highest_score(self):
        counter = 0
        while (
            len(self.entity_score) > 0
            and counter < self.max_entities_from_queue
            and len(self.entities) < self.max_entities_size
        ):
            score, entity = heapq.heappop(self.entity_score)
            if entity not in self.entities:
                self.entities.add(entity)
                counter += 1

    def _update_entities_scores(self, entities):
        for entity in entities.difference(self.entities):
            if entity not in self.entity_coverage:
                self.entity_coverage[entity] = 1
            else:
                self.entity_coverage[entity] += 1
            score = self.entity_coverage[entity] / float(self.graph.get_degree(entity))
            heapq.heappush(self.entity_score, (-score, entity))

    def _add_relation_statements(self, relation_statements):
        for (a, b, sentence) in relation_statements:
            self.entities.update([a, b])
            self.entity_pairs.update([(a, b), (b, a)])
            self.relation_statements[(a, b)] = sentence
            self.ntokens += len(sentence)
            self.nsentences += 1

    def add_initial_entity_pair(self, a, b, max_tokens, max_sentences, sentence):
        a, b = item(a), item(b)
        self.entities.update([a, b])
        self._update_coverage()
        coverage = self.get_coverage(a, b)
        if coverage.num_total_neighbors == 0:
            return False
        return self.try_add_entity_pair_with_neighbors(a, b, max_tokens, max_sentences, 1, sentence)

    def _sample_sentence(self, head_entity, tail_entity):
        edges = self.graph.edges[head_entity].numpy().reshape(-1, GraphDataset.EDGE_SIZE)
        left = np.searchsorted(edges[:, GraphDataset.TAIL_ENTITY], tail_entity, side='left')
        right = np.searchsorted(edges[:, GraphDataset.TAIL_ENTITY], tail_entity, side='right')
        index = np.random.randint(left, right)
        return self.annotated_text.annotate(*edges[index])

    def try_add_entity_pair_with_neighbors(
        self,
        head,
        tail,
        max_tokens,
        max_sentences,
        min_neighbors_to_add,
        sentence=None,
    ):
        head, tail = item(head), item(tail)
        assert head in self.entities and tail in self.entities

        coverage = self.get_coverage(head, tail)
        if coverage.num_total_neighbors == 0:
            return False

        _, neighbors_to_add = coverage.cost()

        num_neighbors_added = len(coverage.both_edges_in_subgraph)
        cur_tokens, cur_sentences = 0, 0
        new_relation_statements = []

        def sample_new_relation_statement(x, y, sentence=None, shall_sample_ordered_pair=True):
            nonlocal cur_sentences
            nonlocal cur_tokens
            if shall_sample_ordered_pair:
                x, y = self._sample_ordered_pair(x, y)
            if sentence is None:
                sentence = self._sample_sentence(x, y)
            new_relation_statements.append((x, y, sentence))
            cur_tokens += len(sentence)
            cur_sentences += 1

        if (head, tail) not in self.entity_pairs:
            sample_new_relation_statement(head, tail, sentence, False)

        for n in neighbors_to_add:
            n_a_exists = (head, n) in self.entity_pairs
            n_b_exists = (n, tail) in self.entity_pairs
            assert not(n_a_exists and n_b_exists)
            if not n_a_exists:
                sample_new_relation_statement(head, n)
            if not n_b_exists:
                sample_new_relation_statement(n, tail)
            num_neighbors_added += 1
            if cur_tokens >= max_tokens or cur_sentences >= max_sentences:
                # NOTE: We keep the last (a, n) and (n, b) entity pairs
                # even though it might to lead to more tokens than max_tokens
                # and/or more sentences than max_sentences.
                if num_neighbors_added < min_neighbors_to_add:
                    # We have failed to add enough edges for the entity pair (a, b).
                    # Thus, we discard all new_relation_statements
                    return False
                else:
                    break

        assert num_neighbors_added >= min_neighbors_to_add
        self._add_relation_statements(new_relation_statements)
        self._update_coverage(new_relation_statements)

        for (a, b, _) in new_relation_statements:
            self._update_entities_scores(self.get_coverage(a, b).both_edges_missing)
        self._add_entities_with_the_highest_score()
        self._update_coverage()
        self.covered_entity_pairs.add((head, tail))
        return True

    def _generate_entity_pair_candidates(self):
        for entity_pair, coverage in self.coverage.items():
            if coverage is None:
                # There is no edge between entity_pair[0] and entity_pair[1]
                continue
            if entity_pair in self.entity_pairs:
                # This entity pair has already been selected
                continue
            yield (entity_pair, coverage)

    def sample_min_cost_entity_pair(self):
        entity_pair_best, cost_to_add_best = [], None

        for entity_pair, coverage in self._generate_entity_pair_candidates():
            current_cost, neighbors_to_add = coverage.cost()
            if cost_to_add_best is None or cost_to_add_best > current_cost:
                entity_pair_best = [entity_pair]
                cost_to_add_best = current_cost
            elif cost_to_add_best == current_cost:
                entity_pair_best.append(entity_pair)

        if len(entity_pair_best) == 0:
            return None, None
        else:
            index = np.random.randint(len(entity_pair_best))
            return entity_pair_best[index], cost_to_add_best

    def _sample_ordered_pair(self, a_or_a_b, b=None):
        if b is None:
            a, b = a_or_a_b
        else:
            a, b = a_or_a_b, b
        return (a, b) if bool(np.random.randint(2, size=1)[0]) else (b, a)

    def fill(self, max_tokens, max_sentences, min_common_neighbors_for_the_last_edge):
        only_accept_zero_cost = False
        while True:
            entity_pair, cost = self.sample_min_cost_entity_pair()
            if entity_pair is None:
                return False

            if only_accept_zero_cost and cost > 0:
                break

            a, b = self._sample_ordered_pair(entity_pair)

            successfully_added = self.try_add_entity_pair_with_neighbors(
                a,
                b,
                max_tokens,
                max_sentences,
                min_common_neighbors_for_the_last_edge,
            )
            if only_accept_zero_cost:
                assert successfully_added

            if (
                not successfully_added
                or self.ntokens >= max_tokens
                or self.nsentences >= max_sentences
            ):
                only_accept_zero_cost = True
        return True

    def relative_coverages(self):
        return np.array([
            self.get_coverage(a, b).relative_coverage()
            for a, b in self.covered_entity_pairs
        ])

    def get_yield(self):
        return len(self.covered_entity_pairs) / float(len(self.relation_statements))

    def get_relative_coverages_mean(self):
        return self.relative_coverages().mean()

    def get_relation_statements(self):
        return self.relation_statements

    def is_covered(self, a_b):
        assert a_b in self.entity_pairs
        return a_b in self.covered_entity_pairs

    def get_covered_edges(self):
        return self.covered_entity_pairs