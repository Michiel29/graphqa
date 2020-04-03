import logging
import numpy as np
import time
import torch

from fairseq.data import FairseqDataset
from utils.plasma_utils import maybe_move_to_plasma
from fairseq.data import data_utils
from utils.data_utils import shuffle_arrays


logger = logging.getLogger(__name__)


class GraphDataset(FairseqDataset):

    # (right_entity, left_entity, left_start_pos, left_end_pos, right_start_pos, right_end_pos, start_block, end_block)
    TAIL_ENTITY = 0
    HEAD_ENTITY = 1
    START_BLOCK = 6
    END_BLOCK = 7

    EDGE_SIZE = 8

    EDGE_CHUNK_SIZE = 3e6

    NEIGHBORS_CACHE_SIZE = 100 * 1024 * 1024 # 100MB

    NUM_THREADS = 5

    def __init__(
        self,
        edges,
        subsampling_strategy,
        subsampling_cap,
        epoch_splits,
        seed,
    ):
        self.edges = edges
        self.subsampling_strategy = subsampling_strategy
        self.subsampling_cap = subsampling_cap
        self.epoch_splits = epoch_splits
        self.epoch_for_generation = None
        self.seed = seed
        self.neighbors_cache = dict()

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.epoch_splits is not None:
            assert epoch >= 1
            epoch_for_generation = (epoch - 1) // self.epoch_splits
            epoch_offset = (epoch - 1) % self.epoch_splits
        else:
            epoch_for_generation = epoch
            epoch_offset = 0

        if epoch_for_generation != self.epoch_for_generation:
            with data_utils.numpy_seed(271828, self.seed, epoch_for_generation):
                indices, sizes = self.subsample_graph_by_entity_pairs()
                start_time = time.time()
                random_permutation = np.random.permutation(len(indices))
                indices = indices[random_permutation]
                sizes = sizes[random_permutation]
                # shuffled edges in 176 seconds
                # shuffle_arrays([indices, sizes])
                logger.info('shuffled edges in %d seconds' % (time.time() - start_time))

                start_time = time.time()
                self._generated_indices = maybe_move_to_plasma(indices)
                self._generated_sizes = maybe_move_to_plasma(sizes)
                logger.info('prepared plasma_arrays in %d seconds' % (time.time() - start_time))

                start_time = time.time()
                self.neighbors_cache = dict()
                entity, total_bytes = 0, 0
                while entity < len(self.edges) and total_bytes < self.NEIGHBORS_CACHE_SIZE:
                    neighbors = self.get_neighbors(entity)
                    total_bytes += neighbors.nelement() * neighbors.element_size()
                    self.neighbors_cache[entity] = neighbors.numpy()
                    entity += 1
                assert entity == len(self.neighbors_cache)
                logger.info('cached neighbours for %d top entities (%.3f MB) in %.3f seconds.' % (
                    entity,
                    total_bytes / 1024 / 1024,
                    time.time() - start_time,
                ))
            self.epoch_for_generation = epoch_for_generation

        start_time = time.time()
        data_per_epoch = len(self._generated_indices.array) // (self.epoch_splits or 1)
        data_start = data_per_epoch * epoch_offset
        data_end = min(len(self._generated_indices.array), data_per_epoch * (epoch_offset + 1))
        self._indices = maybe_move_to_plasma(self._generated_indices.array[data_start:data_end])
        self._sizes = maybe_move_to_plasma(self._generated_sizes.array[data_start:data_end])
        logger.info('selected %d samples from generation epoch %d and epoch offset %d in %d seconds' % (
            data_end - data_start,
            self.epoch_for_generation,
            epoch_offset,
            time.time() - start_time,
        ))

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
        return np.argsort([10 * (np.random.random(len(self.sizes)) - 0.5) + self.sizes])[0]

    def get_neighbors(self, entity):
        if isinstance(entity, torch.Tensor):
            entity = entity.item()
        if entity in self.neighbors_cache:
            return self.neighbors_cache[entity]
        else:
            return self.edges[entity].reshape(-1, self.EDGE_SIZE)[:, self.TAIL_ENTITY].unique()

    def subsample_graph_by_entity_pairs(self):
        from datasets.graph_dataset_util_fast import (
            _count_num_edges_per_head_entity,
            _sample_edges_per_entity_pair,
        )
        start_time = time.time()
        num_entities = len(self.edges)

        num_edges_per_head_entity = np.zeros(num_entities, dtype=np.int32)
        start_entity, start_edge = 0, 0
        chunk_size = None

        while start_entity < num_entities:
            approx_edges_per_entity = int(self.edges._index._sizes[start_entity:start_entity + 5].mean())
            if approx_edges_per_entity > 0:
                chunk_size = max(int(self.EDGE_CHUNK_SIZE / approx_edges_per_entity), 1)
            end_entity = min(num_entities, start_entity + chunk_size)
            head_entities = list(range(start_entity, end_entity))
            head_entities_lens = self.edges._index._sizes[start_entity:end_entity]
            head_entities_pos = np.roll(np.cumsum(head_entities_lens, dtype=np.int64), 1)
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
                num_edges_per_head_entity[start_entity:end_entity],
                self.subsampling_cap,
                self.NUM_THREADS,
            )
            start_edge, start_entity = end_edge, end_entity

        indices = np.zeros((num_edges_per_head_entity.sum(), 2), dtype=np.int64)
        sizes = np.zeros(num_edges_per_head_entity.sum(), dtype=np.int16)

        start_entity, start_edge, start_output_edge = 0, 0, 0
        chunk_size = None
        while start_entity < num_entities:
            approx_edges_per_entity = int(self.edges._index._sizes[start_entity:start_entity + 5].mean())
            if approx_edges_per_entity > 0:
                chunk_size = max(int(self.EDGE_CHUNK_SIZE / approx_edges_per_entity), 1)
            end_entity = min(num_entities, start_entity + chunk_size)
            head_entities = list(range(start_entity, end_entity))

            head_entities_lens = self.edges._index._sizes[start_entity:end_entity]
            head_entities_pos = np.roll(np.cumsum(head_entities_lens, dtype=np.int64), 1)
            head_entities_pos[0] = 0
            end_edge = start_edge + head_entities_lens.sum()

            edges_buffer = np.frombuffer(
                self.edges._bin_buffer,
                dtype=self.edges._index.dtype,
                count=end_edge - start_edge,
                offset=start_edge * self.edges._index.dtype().itemsize,
            )
            output_offsets = np.roll(np.cumsum(num_edges_per_head_entity[start_entity:end_entity], dtype=np.int64), 1)
            output_offsets[0] = 0
            # TODO: Move it out of the loop
            random_scores = np.random.random(len(edges_buffer) // self.EDGE_SIZE)

            end_output_edge = start_output_edge + num_edges_per_head_entity[start_entity:end_entity].sum()
            _sample_edges_per_entity_pair(
                len(head_entities),
                head_entities_pos,
                head_entities_lens,
                edges_buffer,
                sizes[start_output_edge:end_output_edge],
                indices[start_output_edge:end_output_edge],
                output_offsets,
                random_scores,
                self.subsampling_cap,
                self.NUM_THREADS,
            )
            indices[start_output_edge:end_output_edge, 0] += head_entities[0]
            start_edge, start_entity, start_output_edge = end_edge, end_entity, end_output_edge

        logger.info('subsampled %d edges by entity pairs (cap = %d) in %.3f seconds.' % (
            len(sizes),
            self.subsampling_cap,
            time.time() - start_time,
        ))

        return indices, sizes