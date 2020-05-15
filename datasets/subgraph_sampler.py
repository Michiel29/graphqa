from collections import namedtuple
import heapq
from intervaltree import IntervalTree
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


RelationStatement = namedtuple('RelationStatement', ['head', 'tail', 'sentence', 'begin', 'end'])


class SubgraphSampler(object):

    def __init__(
        self,
        graph,
        annotated_text,
        min_common_neighbors,
        max_entities_size,
        max_entities_from_queue,
        cover_random_prob,
    ):
        self.graph = graph
        self.annotated_text = annotated_text
        self.min_common_neighbors = min_common_neighbors
        self.max_entities_size = max_entities_size
        self.max_entities_from_queue = max_entities_from_queue
        self.cover_random_prob = cover_random_prob
        self.entities = set([])
        self.entity_pairs = set([])
        self.covered_entity_pairs = set([])
        self.coverage = {}
        self.relation_statements = {}
        self.ntokens = 0
        self.nsentences = 0

        self.entity_score = []
        self.entity_coverage = {}
        self.intervals = IntervalTree()

    def _update_coverage(self, new_relation_statements=None):
        new_relation_statements_tmp = None
        if new_relation_statements is not None:
            new_relation_statements_tmp = [(rs.head, rs.tail) for rs in new_relation_statements]
        update_coverage(
            self.graph,
            self.entities,
            self.entity_pairs,
            self.coverage,
            self.min_common_neighbors,
            new_relation_statements_tmp,
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
        for rs in relation_statements:
            self.entities.update([rs.head, rs.tail])
            self.entity_pairs.update([(rs.head, rs.tail), (rs.tail, rs.head)])
            self.relation_statements[(rs.head, rs.tail)] = rs.sentence
            assert len(self.intervals.overlap(rs.begin, rs.end)) == 0
            self.intervals.addi(rs.begin, rs.end, None)
            self.ntokens += len(rs.sentence)
            self.nsentences += 1

    def add_initial_entity_pair(self, a, b, max_tokens, max_sentences, sentence):
        a, b = item(a), item(b)
        self.entities.update([a, b])
        self._update_coverage()
        coverage = self.get_coverage(a, b)
        if coverage.num_total_neighbors == 0:
            return False
        return self.try_add_entity_pair_with_neighbors(a, b, max_tokens, max_sentences, 1)

    def intervals_overlap(self, i1, i2):
        if i1[0] <= i2[0] and i2[0] < i1[1]:
            return True
        if i2[0] <= i1[0] and i1[0] < i2[1]:
            return True
        return False

    def _sample_relation_statement(self, head_entity, tail_entity, local_intervals, local_interval=None):
        edges = self.graph.edges[head_entity].numpy().reshape(-1, GraphDataset.EDGE_SIZE)
        left = np.searchsorted(edges[:, GraphDataset.TAIL_ENTITY], tail_entity, side='left')
        right = np.searchsorted(edges[:, GraphDataset.TAIL_ENTITY], tail_entity, side='right')
        indices = np.arange(left, right)
        np.random.shuffle(indices)

        at_least_a_single_edge_exists = False
        for index in indices:
            begin = edges[index][GraphDataset.START_BLOCK]
            end = edges[index][GraphDataset.END_BLOCK]
            if len(self.intervals.overlap(begin, end)) > 0:
                continue
            at_least_a_single_edge_exists = True
            if (
                len(local_intervals.overlap(begin, end)) == 0
                and (local_interval is None or not self.intervals_overlap((begin, end), local_interval))
            ):
                return RelationStatement(
                    head=head_entity,
                    tail=tail_entity,
                    sentence=self.annotated_text.annotate(*edges[index]),
                    begin=begin,
                    end=end,
                )

        # TODO: Clean up some of the edges
        # if not at_least_a_single_edge_exists:
        #     # None of the edges between head and tail entities can be used.
        #     # Therefore, we might as well drop their connection
        #     x, y = min(head_entity, tail_entity), max(head_entity, tail_entity)
        #     if (x, y) in self.coverage and not ((x, y) in self.relation_statements or (y, x) in self.relation_statements):
        #         self.coverage[(x, y)] = None
        return None

    def try_add_entity_pair_with_neighbors(
        self,
        head,
        tail,
        max_tokens,
        max_sentences,
        min_neighbors_to_add,
    ):
        head, tail = item(head), item(tail)
        assert head in self.entities and tail in self.entities

        coverage = self.get_coverage(head, tail)
        if coverage.num_total_neighbors == 0:
            return False
        num_neighbors_added = len(coverage.both_edges_in_subgraph)

        cur_tokens, cur_sentences = 0, 0
        new_relation_statements = []
        local_intervals = IntervalTree()

        def sample_new_relation_statements(edge1, edge2=None):
            nonlocal cur_sentences
            nonlocal cur_tokens
            nonlocal local_intervals
            x1, y1 = self._sample_ordered_pair(*edge1)
            rs1 = self._sample_relation_statement(x1, y1, local_intervals)
            if rs1 is None:
                return False
            if edge2 is not None:
                x2, y2 = self._sample_ordered_pair(*edge2)
                rs2 = self._sample_relation_statement(x2, y2, local_intervals, (rs1.begin, rs1.end))
                if rs2 is None:
                    return False
            new_relation_statements.append(rs1)
            cur_tokens += len(rs1.sentence)
            cur_sentences += 1
            local_intervals.addi(rs1.begin, rs1.end, None)
            if edge2 is not None:
                new_relation_statements.append(rs2)
                cur_tokens += len(rs2.sentence)
                cur_sentences += 1
                local_intervals.addi(rs2.begin, rs2.end, None)
            return True

        if (head, tail) not in self.entity_pairs:
            if not sample_new_relation_statements((head, tail)):
                return False

        if num_neighbors_added < self.min_common_neighbors:
            for n in self._generate_neighbors_to_add(head, tail):
                n_a_exists = (head, n) in self.entity_pairs
                n_b_exists = (n, tail) in self.entity_pairs
                assert not(n_a_exists and n_b_exists)
                if not n_a_exists and not n_b_exists:
                    result = sample_new_relation_statements((head, n), (n, tail))
                elif not n_a_exists:
                    result = sample_new_relation_statements((head, n))
                elif not n_b_exists:
                    result = sample_new_relation_statements((n, tail))
                else:
                    raise Exception('Impossible state')
                if not result:
                    continue

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

                if num_neighbors_added >= self.min_common_neighbors:
                    break

        if num_neighbors_added < min_neighbors_to_add:
            return False

        self._add_relation_statements(new_relation_statements)
        self._update_coverage(new_relation_statements)

        for rs in new_relation_statements:
            coverage = self.get_coverage(rs.head, rs.tail)
            if coverage is not None:
                self._update_entities_scores(coverage.both_edges_missing)
        self._add_entities_with_the_highest_score()
        self._update_coverage()
        if (head, tail) in self.relation_statements:
            self.covered_entity_pairs.add((head, tail))
        else:
            self.covered_entity_pairs.add((tail, head))
        return True

    def _generate_entity_pair_candidates(self):
        for entity_pair, coverage in self.coverage.items():
            if coverage is None:
                # There is no edge between entity_pair[0] and entity_pair[1]
                continue
            if entity_pair in self.covered_entity_pairs:
                # This entity pair has already been selected
                continue
            yield (entity_pair, coverage)

    def sample_min_cost_entity_pair(self):
        entity_pair_best, cost_to_add_best = [], None

        cover_random = bool(np.random.binomial(1, self.cover_random_prob))
        if cover_random:
            all_pairs = list(self._generate_entity_pair_candidates())
            if len(all_pairs) == 0:
                return None, None
            entity_pair_idx = np.random.randint(len(all_pairs))
            entity_pair_random = all_pairs[entity_pair_idx][0]
            cost_to_add_random = all_pairs[entity_pair_idx][1].cost()
            return entity_pair_random, cost_to_add_random

        for entity_pair, coverage in self._generate_entity_pair_candidates():
            current_cost = coverage.cost()
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

    def _generate_neighbors_to_add(self, a, b):
        coverage = self.get_coverage(a, b)
        neighbors = np.array([x for x in coverage.single_edge_missing], dtype=np.int64)
        np.random.shuffle(neighbors)
        for n in neighbors:
            yield n
        neighbors = np.array([x for x in coverage.both_edges_missing], dtype=np.int64)
        np.random.shuffle(neighbors)
        for n in neighbors:
            yield n

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

            successfully_added = self.try_add_entity_pair_with_neighbors(
                entity_pair[0],
                entity_pair[1],
                max_tokens,
                max_sentences,
                min_common_neighbors_for_the_last_edge,
            )

            # if only_accept_zero_cost:
            #     assert successfully_added

            if not successfully_added:
                assert entity_pair in self.coverage
                # We cannot consider this entity pair as possible target edge
                self.coverage[entity_pair] = None

            if self.ntokens >= max_tokens or self.nsentences >= max_sentences:
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