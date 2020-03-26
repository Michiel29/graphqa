import numpy as np
import cython
from cython.parallel import prange
from libcpp.queue cimport priority_queue
from libcpp.pair cimport pair

cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void _count_num_edges_per_head_entity(
    int num_entities,
    const int[:] pos,
    const int[:] sizes,
    const int[:] edges_buffer,
    int[:] output,
    int subsampling_cap,
    int nthreads
):
    cdef int head_entity, tail_entity
    cdef int head_entity_start, head_entity_end

    cdef int edge_idx
    cdef int num_edges_per_entity_pair
    cdef int previous_tail_entity

    for head_entity in prange(num_entities, nogil=True, num_threads=nthreads):
        if sizes[head_entity] == 0:
            continue

        head_entity_start = pos[head_entity]
        head_entity_end = head_entity_start + sizes[head_entity]
        previous_tail_entity = -1
        num_edges_per_entity_pair = 0

        for edge_idx in range(head_entity_start, head_entity_end, 8):
            tail_entity = edges_buffer[edge_idx]
            if tail_entity != previous_tail_entity:
                if num_edges_per_entity_pair > subsampling_cap:
                    output[head_entity] = output[head_entity] + subsampling_cap
                else:
                    output[head_entity] = output[head_entity] + num_edges_per_entity_pair
                num_edges_per_entity_pair = 1
                previous_tail_entity = tail_entity
            else:
                num_edges_per_entity_pair = num_edges_per_entity_pair + 1

        if num_edges_per_entity_pair > subsampling_cap:
            output[head_entity] = output[head_entity] + subsampling_cap
        else:
            output[head_entity] = output[head_entity] + num_edges_per_entity_pair


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void _sample_edges_per_entity_pair(
    int num_entities,
    const int[:] pos,
    const int[:] sizes,
    const int[:] edges_buffer,
    int[:] output_lens,
    int[:, :] output,
    const int[:] output_offsets,
    const double[:] scores,
    int subsampling_cap,
    int nthreads
):
    cdef int head_entity, tail_entity
    cdef int head_entity_start, head_entity_end

    cdef int edge_idx
    cdef int edges_per_entity_counter, counter
    cdef int previous_tail_entity, tail_entity_edges_start
    cdef pair[double, int] sampled_entity_pair
    cdef int actual_edge_idx, output_index
    cdef priority_queue[pair[double, int]] entity_pair_scores

    for head_entity in prange(num_entities, nogil=True, num_threads=nthreads):
        if sizes[head_entity] == 0:
            continue

        head_entity_start = pos[head_entity]
        head_entity_end = head_entity_start + sizes[head_entity]
        previous_tail_entity = -1
        entity_pair_scores = priority_queue[pair[double, int]]()
        output_index = 0

        for edge_idx in range(head_entity_start, head_entity_end, 8):
            tail_entity = edges_buffer[edge_idx]
            if tail_entity != previous_tail_entity:
                counter = 0
                while counter < subsampling_cap and entity_pair_scores.size() > 0:
                    sampled_entity_pair = entity_pair_scores.top()
                    entity_pair_scores.pop()
                    actual_edge_idx = int((sampled_entity_pair.second - head_entity_start) / 8)
                    output[output_offsets[head_entity] + output_index][0] = head_entity
                    output[output_offsets[head_entity] + output_index][1] = actual_edge_idx
                    output_lens[output_offsets[head_entity] + output_index] = (
                        edges_buffer[sampled_entity_pair.second + 7]
                        - edges_buffer[sampled_entity_pair.second + 6]
                    )
                    counter = counter + 1
                    output_index = output_index + 1
                previous_tail_entity = tail_entity
                entity_pair_scores = priority_queue[pair[double, int]]()

            actual_edge_idx = int(edge_idx / 8)
            entity_pair_scores.push(pair[double, int](scores[actual_edge_idx], edge_idx))

        counter = 0
        while counter < subsampling_cap and entity_pair_scores.size() > 0:
            sampled_entity_pair = entity_pair_scores.top()
            entity_pair_scores.pop()
            actual_edge_idx = int((sampled_entity_pair.second - head_entity_start) / 8)
            output[output_offsets[head_entity] + output_index][0] = head_entity
            output[output_offsets[head_entity] + output_index][1] = actual_edge_idx
            output_lens[output_offsets[head_entity] + output_index] = (
                edges_buffer[sampled_entity_pair.second + 7]
                - edges_buffer[sampled_entity_pair.second + 6]
            )
            counter = counter + 1
            output_index = output_index + 1