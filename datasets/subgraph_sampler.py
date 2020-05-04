import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import logging


logger = logging.getLogger(__name__)


def item(tensor):
    if isinstance(tensor, torch.Tensor):
        assert tensor.numel() == 1
        return tensor.item()
    else:
        return tensor


class NeighborhoodCoverage(object):

    INF_COST = 1e9

    def __init__(
        self,
        head,
        tail,
        common_neighbors,
        entity_pairs_in_subgraph,
        min_common_neighbors,
    ):
        self.head = head
        self.tail = tail
        self.min_common_neighbors = min_common_neighbors
        self.head_tail_set = set([head, tail])
        self.num_total_neighbors = len(common_neighbors)
        self.head_tail_in_subgraph = (head, tail) in entity_pairs_in_subgraph

        self.both_edges_in_subgraph = set()
        self.single_edge_missing = set()
        self.both_edges_missing = set()
        for n in common_neighbors:
            n_head_exists = (head, n) in entity_pairs_in_subgraph
            n_tail_exists = (tail, n) in entity_pairs_in_subgraph
            if n_head_exists and n_tail_exists:
                self.both_edges_in_subgraph.add(n)
            elif not n_head_exists and not n_tail_exists:
                self.both_edges_missing.add(n)
            else:
                self.single_edge_missing.add(n)
        self.reset_cost()

    def _update_neighbor(self, n):
        if n in self.both_edges_missing:
            self.both_edges_missing.remove(n)
            self.single_edge_missing.add(n)
        elif n in self.single_edge_missing:
            self.single_edge_missing.remove(n)
            self.both_edges_in_subgraph.add(n)
        elif n in self.both_edges_in_subgraph:
            raise Exception('Both edges %d-%d-%d already exist.' % (self.head, n, self.tail))
        else:
            # This is the case when n has an edge with only either head or tail, but not both.
            # Thus, adding this edge has no effect on the neighborhood coverage for (head, tail).
            # So we don't need to relaculate the cost.
            return
        self.reset_cost()

    def add_entity_pair(self, a_b_set):
        if len(self.head_tail_set.intersection(a_b_set)) == 2:
            self.head_tail_in_subgraph = True
        elif len(self.head_tail_set.intersection(a_b_set)) == 1:
            n = a_b_set.difference(self.head_tail_set)
            assert len(n) == 1
            self._update_neighbor(next(iter(n)))

    def _compute_cost(self):
        if self.num_total_neighbors == 0:
            return NeighborhoodCoverage.INF_COST, []

        target_num_of_common_neighbors = min(self.min_common_neighbors, self.num_total_neighbors)
        num_common_neighbors_to_cover = max(
            target_num_of_common_neighbors - len(self.both_edges_in_subgraph),
            0,
        )

        if num_common_neighbors_to_cover == 0:
            return int(self.head_tail_in_subgraph), []
        else:
            neighbors_to_add = list(self.single_edge_missing)
            cost_to_add = len(self.single_edge_missing)
            if len(neighbors_to_add) > num_common_neighbors_to_cover:
                neighbors_to_add = np.random.choice(
                    neighbors_to_add,
                    num_common_neighbors_to_cover,
                    replace=False,
                )
                cost_to_add = num_common_neighbors_to_cover
            else:
                neighbors_to_add_tmp = np.random.choice(
                    list(self.both_edges_missing),
                    num_common_neighbors_to_cover - len(neighbors_to_add),
                    replace=False,
                )
                cost_to_add += 2 * len(neighbors_to_add_tmp)
                neighbors_to_add.extend(neighbors_to_add_tmp)
            return cost_to_add + int(self.head_tail_in_subgraph), neighbors_to_add

    def reset_cost(self):
        self._cost = None

    def cost(self):
        if self._cost is None:
            self._cost = self._compute_cost()
        return self._cost

    def relative_coverage(self):
        if self.num_total_neighbors == 0:
            return 1
        return len(self.both_edges_in_subgraph) / float(self.num_total_neighbors)


class SubgraphSampler(object):
    def __init__(self, graph, annotated_text, min_common_neighbors):
        self.graph = graph
        self.annotated_text = annotated_text
        self.min_common_neighbors = min_common_neighbors
        self.entities = set([])
        self.entity_pairs = set([])
        self.covered_entity_pairs = set([])
        self.coverage = {}
        self.relation_statements = {}
        self.ntokens = 0
        self.nsentences = 0

    def _update_coverage(self, new_entity_pair=None):
        for a in self.entities:
            a_neighbors = None
            for b in self.entities:
                if a >= b:
                    continue
                if (a, b) not in self.coverage:
                    if a_neighbors is None:
                        a_neighbors = self.graph.get_neighbors(a)
                    if b not in a_neighbors:
                        # There is no edge between a and b, so it's pointless
                        # to compute common neighbors for them
                        self.coverage[(a, b)] = None
                    else:
                        b_neighbors = self.graph.get_neighbors(b)
                        a_b_neighbors = np.intersect1d(
                            a_neighbors,
                            b_neighbors,
                            assume_unique=True,
                        )
                        self.coverage[(a, b)] = NeighborhoodCoverage(
                            head=a,
                            tail=b,
                            common_neighbors=a_b_neighbors,
                            entity_pairs_in_subgraph=self.entity_pairs,
                            min_common_neighbors=self.min_common_neighbors,
                        )
                elif self.coverage[(a, b)] is not None and new_entity_pair is not None:
                    self.coverage[(a, b)].add_entity_pair(set(new_entity_pair))

    def get_coverage(self, a, b):
        return self.coverage[(min(a, b), max(a, b))]

    def _add_entity_pair(self, a, b, sentence):
        assert (a, b) not in self.entity_pairs
        self.entities.update([a, b])
        self.entities.update(self.get_coverage(a, b).both_edges_missing)
        self.entity_pairs.update([(a, b), (b, a)])
        self.relation_statements[(a, b)] = sentence
        self.ntokens += len(sentence)
        self.nsentences += 1
        self._update_coverage((a, b))

    def try_add_entity_pair_with_neighbors(
        self,
        a,
        b,
        max_tokens,
        max_sentences,
        min_neighbors_to_add,
        sentence=None,
    ):
        a, b = item(a), item(b)

        if a not in self.entities or b not in self.entities:
            self.entities.update([a, b])
            self._update_coverage()
        coverage = self.get_coverage(a, b)
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
                # TODO: sample sentence
                sentence = torch.ones(100)
            new_relation_statements.append((x, y, sentence))
            cur_tokens += len(sentence)
            cur_sentences += 1

        if (a, b) not in self.entity_pairs:
            sample_new_relation_statement(a, b, sentence, False)

        for n in neighbors_to_add:
            n_a_exists = (a, n) in self.entity_pairs
            n_b_exists = (b, n) in self.entity_pairs
            assert not(n_a_exists and n_b_exists)
            if not n_a_exists:
                sample_new_relation_statement(a, n)
            if not n_b_exists:
                sample_new_relation_statement(b, n)
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
        for (na, nb, sentence) in new_relation_statements:
            self._add_entity_pair(na, nb, sentence)
        self.covered_entity_pairs.add((a, b))
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
