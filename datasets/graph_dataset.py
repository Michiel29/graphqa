import logging
import numpy as np
import time

from fairseq.data import FairseqDataset, plasma_utils
from fairseq.data import data_utils


logger = logging.getLogger(__name__)


class GraphDataset(FairseqDataset):

    # (right_entity, left_entity, left_start_pos, left_end_pos, right_start_pos, right_end_pos, start_block, end_block)
    TAIL_ENTITY = 0
    HEAD_ENTITY = 1
    START_BLOCK = 6
    END_BLOCK = 7

    EDGE_SIZE = 8

    EDGE_CHUNK_SIZE = 3e6

    NUM_THREADS = 5

    def __init__(self, edges, subsampling_strategy, subsampling_cap, seed):
        self.edges = edges
        self.seed = seed
        self.subsampling_strategy = subsampling_strategy
        self.subsampling_cap = subsampling_cap
        self.epoch = None
        self.indices = None
        # self.set_epoch(0)
        self._sizes = None

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            with data_utils.numpy_seed(271828, self.seed, epoch):
                self.subsample_graph_by_entity_pairs()
            self.epoch = epoch

    def __getitem__(self, index):
        source_entity, local_index = self._indices.array[index]
        edges = self.edges[source_entity].reshape(-1, 8)
        return edges[local_index]

    def __len__(self):
        return len(self._indices.array)

    def num_tokens(self, index):
        return self._sizes.array[index]

    def size(self, index):
        return self._sizes.array[index]

    @property
    def sizes(self):
        return self._sizes.array

    def ordered_indices(self):
        return np.lexsort([
            np.random.permutation(len(self)),
            self.sizes,
        ])

    def get_neighbors(self, entity):
        return self.edges[entity].reshape(-1, self.EDGE_SIZE)[:, self.TAIL_ENTITY].unique()

    def subsample_graph_by_entity_pairs(self):
        from datasets.graph_dataset_util_fast import (
            _count_num_edges_per_head_entity,
            _sample_edges_per_entity_pair,
        )
        start_time = time.time()
        start_entity, start_edge = 0, 0
        chunk_size = None
        indices, sizes = list(), list()

        while start_entity < len(self.edges):
            approx_edges_per_entity = int(self.edges._index._sizes[start_entity:start_entity + 5].mean())
            if approx_edges_per_entity > 0:
                chunk_size = int(self.EDGE_CHUNK_SIZE / approx_edges_per_entity)
            end_entity = min(len(self.edges), start_entity + chunk_size)
            head_entities = list(range(start_entity, end_entity))
            num_edges_per_head_entity = np.zeros(end_entity - start_entity, dtype=np.int32)

            head_entities_lens = self.edges._index._sizes[start_entity:end_entity]
            head_entities_pos = np.roll(np.cumsum(head_entities_lens, dtype=np.int32), 1)
            head_entities_pos[0] = 0
            end_edge = start_edge + head_entities_lens.sum()

            edges_buffer = np.frombuffer(
                self.edges._bin_buffer,
                dtype=self.edges._index.dtype,
                count=end_edge - start_edge,
                offset=start_edge * self.edges._index.dtype().itemsize,
            )
            _count_num_edges_per_head_entity(
                len(head_entities),
                head_entities_pos,
                head_entities_lens,
                edges_buffer,
                num_edges_per_head_entity,
                self.subsampling_cap,
                self.NUM_THREADS,
            )
            chunk_sizes = np.zeros(num_edges_per_head_entity.sum(), dtype=np.int32)
            chunk_indices = np.zeros((num_edges_per_head_entity.sum(), 2), dtype=np.int32)
            output_offsets = np.roll(np.cumsum(num_edges_per_head_entity, dtype=np.int32), 1)
            output_offsets[0] = 0
            random_scores = np.random.random(len(edges_buffer) // self.EDGE_SIZE)
            _sample_edges_per_entity_pair(
                len(head_entities),
                head_entities_pos,
                head_entities_lens,
                edges_buffer,
                chunk_sizes,
                chunk_indices,
                output_offsets,
                random_scores,
                self.subsampling_cap,
                self.NUM_THREADS,
            )
            chunk_indices[:, 0] += head_entities[0]
            indices.append(chunk_indices)
            sizes.append(chunk_sizes)
            start_edge, start_entity = end_edge, end_entity

        self._indices = plasma_utils.PlasmaArray(np.concatenate(indices))
        self._sizes = plasma_utils.PlasmaArray(np.concatenate(sizes))
        logger.info(
            'subsample graph by entity pairs: graph subsampled in %.3f seconds.' % (
            time.time() - start_time,
        ))