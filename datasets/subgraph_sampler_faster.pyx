import numpy as np
import cython
from libcpp cimport bool

cimport numpy as np
cimport cython


ctypedef np.int64_t DTYPE_t


cdef class NeighborhoodCoverage(object):

    cdef public int INF_COST

    cdef public int head
    cdef public int tail
    cdef public int num_total_neighbors
    cdef public int min_common_neighbors

    cdef public bool head_tail_in_subgraph

    cdef public set head_tail_set
    cdef public set both_edges_in_subgraph
    cdef public set single_edge_missing
    cdef public set both_edges_missing

    cdef int _cost

    def __init__(
        self,
        int head,
        int tail,
        np.ndarray[DTYPE_t, ndim=1] head_neighbors,
        np.ndarray[DTYPE_t, ndim=1] tail_neighbors,
        set entity_pairs_in_subgraph,
        int min_common_neighbors,
    ):
        self.INF_COST = 100000000
        self.head = head
        self.tail = tail
        self.min_common_neighbors = min_common_neighbors

        self.head_tail_set = set([head, tail])
        self.both_edges_in_subgraph = set()
        self.single_edge_missing = set()
        self.both_edges_missing = set()

        self.head_tail_in_subgraph = (min(self.head, self.tail), max(self.head, self.tail)) in entity_pairs_in_subgraph

        self._update_neighbors_sets(head_neighbors, tail_neighbors, entity_pairs_in_subgraph)
        self._reset_cost()

    cdef _update_neighbors_sets(
        self,
        np.ndarray[DTYPE_t, ndim=1] head_neighbors,
        np.ndarray[DTYPE_t, ndim=1] tail_neighbors,
        set entity_pairs_in_subgraph,
    ):
        cdef int n
        cdef int head_index = 0
        cdef int head_length = head_neighbors.size
        cdef int tail_index = 0
        cdef int tail_length = tail_neighbors.size
        cdef bool n_head_exists, n_tail_exists

        self.num_total_neighbors = 0

        while (head_index < head_length) and (tail_index < tail_length):
            if head_index > 0:
                assert head_neighbors[head_index - 1] < head_neighbors[head_index]
            if tail_index > 0:
                assert tail_neighbors[tail_index - 1] < tail_neighbors[tail_index]

            if head_neighbors[head_index] < tail_neighbors[tail_index]:
                head_index += 1
                continue
            if head_neighbors[head_index] > tail_neighbors[tail_index]:
                tail_index += 1
                continue

            n = head_neighbors[head_index]
            self.num_total_neighbors += 1
            head_index += 1
            tail_index += 1

            n_head_exists = (self.head, n) in entity_pairs_in_subgraph
            n_tail_exists = (self.tail, n) in entity_pairs_in_subgraph

            if n_head_exists and n_tail_exists:
                self.both_edges_in_subgraph.add(n)
            elif not n_head_exists and not n_tail_exists:
                self.both_edges_missing.add(n)
            else:
                self.single_edge_missing.add(n)

    cdef _reset_cost(self):
        self._cost = -1

    cdef _update_neighbor(self, n):
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
            self._reset_cost()

    def add_entity_pair(self, a_b_set):
        if len(self.head_tail_set.intersection(a_b_set)) == 2:
            self.head_tail_in_subgraph = True
        elif len(self.head_tail_set.intersection(a_b_set)) == 1:
            n = a_b_set.difference(self.head_tail_set)
            assert len(n) == 1
            self._update_neighbor(next(iter(n)))

    cdef _compute_cost(self):
        cdef int target_num_of_common_neighbors
        cdef int num_common_neighbors_to_cover

        if self.num_total_neighbors == 0:
            self._cost = self.INF_COST
            return

        target_num_of_common_neighbors = min(self.min_common_neighbors, self.num_total_neighbors)
        num_common_neighbors_to_cover = max(
            target_num_of_common_neighbors - len(self.both_edges_in_subgraph),
            0,
        )

        if num_common_neighbors_to_cover > 0:
            if len(self.single_edge_missing) > num_common_neighbors_to_cover:
                self._cost = num_common_neighbors_to_cover
            else:
                self._cost = len(self.single_edge_missing)
                neighbors_to_add_tmp = num_common_neighbors_to_cover - len(self.single_edge_missing)
                self._cost += 2 * neighbors_to_add_tmp

        self._cost += int(self.head_tail_in_subgraph)

    def cost(self):
        if self._cost == -1:
            self._compute_cost()
        return self._cost

    def relative_coverage(self):
        if self.num_total_neighbors == 0:
            return 1
        return len(self.both_edges_in_subgraph) / float(self.num_total_neighbors)


def update_coverage(
    graph,
    set entities,
    set entity_pairs,
    dict coverage_dict,
    int min_common_neighbors,
    list new_relation_statements=None,
):
    for a in entities:
        a_neighbors = None
        for b in entities:
            if a >= b:
                continue
            if (a, b) not in coverage_dict:
                if a_neighbors is None:
                    a_neighbors = graph.get_neighbors(a)
                if b not in a_neighbors:
                    # There is no edge between a and b, so it's pointless
                    # to compute common neighbors for them
                    coverage_dict[(a, b)] = None
                else:
                    b_neighbors = graph.get_neighbors(b)
                    coverage_dict[(a, b)] = NeighborhoodCoverage(
                        head=a,
                        tail=b,
                        head_neighbors=a_neighbors,
                        tail_neighbors=b_neighbors,
                        entity_pairs_in_subgraph=entity_pairs,
                        min_common_neighbors=min_common_neighbors,
                    )
            elif coverage_dict[(a, b)] is not None and new_relation_statements is not None:
                for new_a, new_b in new_relation_statements:
                    coverage_dict[(a, b)].add_entity_pair(set([new_a, new_b]))