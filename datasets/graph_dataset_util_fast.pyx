import numpy as np
import cython
from cython.parallel import prange

cimport numpy as np
cimport cython


cdef inline int int_min(int a, int b): return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[np.int32_t, ndim=1] _count_num_edges_per_head_entity(
    int num_entities,
    const int[:] edges_head_entity_starts,
    const int[:] edges_head_entity_lens,
    const int[:] edges_buffer,
    int subsampling_cap,
    int nthreads
):
    cdef int buffer_len = edges_buffer.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] num_edges_per_head_entity = np.zeros(num_entities, np.int32)

    cdef int start_entity = edges_buffer[1]
    cdef int head_entity, tail_entity
    cdef int edge_idx
    cdef int num_edges_per_entity_pair = 0
    cdef int previous_tail_entity = -1
    cdef int head_entity_start, head_entity_end

    assert buffer_len % 8 == 0

    for head_entity in range(num_entities):
        head_entity_start = edges_head_entity_starts[head_entity]
        head_entity_end = head_entity_start + edges_head_entity_lens[head_entity]
        previous_tail_entity = -1
        for edge_idx in range(head_entity_start, head_entity_end, 8):
            assert edges_buffer[edge_idx + 1] == start_entity + head_entity
            tail_entity = edges_buffer[edge_idx]
            if tail_entity != previous_tail_entity:
                if num_edges_per_entity_pair > subsampling_cap:
                    num_edges_per_head_entity[head_entity] += subsampling_cap
                else:
                    num_edges_per_head_entity[head_entity] += num_edges_per_entity_pair
                num_edges_per_entity_pair = 1
                previous_tail_entity = tail_entity
            else:
                num_edges_per_entity_pair += 1
        if num_edges_per_entity_pair > subsampling_cap:
            num_edges_per_head_entity[head_entity] += subsampling_cap
        else:
            num_edges_per_head_entity[head_entity] += num_edges_per_entity_pair
    return num_edges_per_head_entity
