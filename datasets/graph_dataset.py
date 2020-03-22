import logging
import numpy as np
import time

from fairseq.data import FairseqDataset, plasma_utils


logger = logging.getLogger(__name__)


class GraphDataset(FairseqDataset):

    # (right_entity, left_entity, left_start_pos, left_end_pos, right_start_pos, right_end_pos, start_block, end_block)
    TAIL_ENTITY = 0
    HEAD_ENTITY = 1
    START_BLOCK = 6
    END_BLOCK = 7

    EDGE_SIZE = 8

    EDGE_CHUNK_SIZE = 3276800

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
        self.epoch = epoch
        self.subsample_graph_by_entity_pairs()
        # perform subsampling
        # _sizes, _indices = list(), list()
        # blabla_counter = 0
        # for source_entity in range(len(self.edges)):
        #     edges = self.edges[source_entity].reshape(-1, 8)
        #     num_edges = edges.shape[0]
        #     _sizes.append(edges[:, GraphDataset.END_BLOCK] - edges[:, GraphDataset.START_BLOCK])
        #     _indices.append(np.stack([np.full(num_edges, source_entity), np.arange(num_edges)], axis=1))
        #     blabla_counter += 1
        #     if blabla_counter == 10000:
        #         break
        # self._sizes = np.concatenate(_sizes)
        # self._indices = np.concatenate(_indices)

    def __getitem__(self, index):
        source_entity, local_index = self._indices[index]
        edges = self.edges[source_entity].reshape(-1, 8)
        return edges[local_index]

    def __len__(self):
        return len(self._indices)

    def num_tokens(self, index):
        return self._sizes[index]

    def size(self, index):
        return self._sizes[index]

    @property
    def sizes(self):
        return self._sizes

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
        indices = list()

        while start_entity < len(self.edges):
            approx_edges_per_entity = int(self.edges._index._sizes[start_entity:start_entity + 5].mean())
            if approx_edges_per_entity > 0:
                chunk_size = self.EDGE_CHUNK_SIZE // approx_edges_per_entity
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
                10,
            )
            chunk_indices = np.zeros((num_edges_per_head_entity.sum(), 2), dtype=np.int32)
            output_offsets = np.roll(np.cumsum(num_edges_per_head_entity, dtype=np.int32), 1)
            output_offsets[0] = 0
            random_scores = np.random.random(num_edges_per_head_entity.sum())
            _sample_edges_per_entity_pair(
                len(head_entities),
                head_entities_pos,
                head_entities_lens,
                edges_buffer,
                chunk_indices,
                output_offsets,
                random_scores,
                self.subsampling_cap,
                10,
            )
            chunk_indices[:, 0] += head_entities[0]
            indices.append(chunk_indices)
            start_edge, start_entity = end_edge, end_entity
        # plasma_utils.PlasmaArray(slice_indices)
        logger.info(
            'subsample graph by entity pairs: graph subsampled in %.3f seconds.' % (
            time.time() - start_time,
        ))
        x = 1