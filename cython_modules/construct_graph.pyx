import numpy as np
cimport numpy as np

from collections import defaultdict
from itertools import combinations

def construct_graph(annotation_data, n_entities):
    neighbor_list = [list() for entity in range(n_entities)]
    edge_dict = defaultdict(list)

    cdef int sentence_idx
    cdef np.ndarray[np.int64_t, ndim=1] entity_ids

    cdef int a
    cdef int b

    for sentence_idx in range(len(annotation_data)):
        entity_ids = annotation_data[sentence_idx].reshape(-1, 3)[:, -1].numpy()

        for a, b in combinations(entity_ids, 2):
            neighbor_list[a].append(b)
            neighbor_list[b].append(a)

        edge_dict[frozenset({a, b})].append(sentence_idx)

    return neighbor_list, edge_dict
