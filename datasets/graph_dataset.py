import logging
import numpy as np
import time

from fairseq.data import FairseqDataset


logger = logging.getLogger(__name__)


class GraphDataset(FairseqDataset):

    # (right_entity, left_entity, left_start_pos, left_end_pos, right_start_pos, right_end_pos, start_block, end_block)
    TAIL_ENTITY = 0
    HEAD_ENTITY = 1
    START_BLOCK = 6
    END_BLOCK = 7

    EDGE_SIZE = 8

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
        if not hasattr(self, "num_edges_per_head_entity"):
            from datasets.graph_dataset_util_fast import (
                _count_num_edges_per_head_entity,
            )
            start_time = time.time()

            chunk_size = 1000
            self.num_edges_per_head_entity = np.zeros(len(self.edges), dtype=np.int32)
            start_block = 0
            for start_chunk in range(0, len(self.edges), chunk_size):
                head_entities = list(range(start_chunk, min(len(self.edges), start_chunk + chunk_size)))
                head_entities_lens = np.array([len(self.edges[head_entity]) for head_entity in head_entities], dtype=np.int32)
                head_entities_pos = np.roll(np.cumsum(head_entities_lens, dtype=np.int32), 1)
                head_entities_pos[0] = 0
                end_block = start_block + head_entities_lens.sum()
                edges_buffer = np.frombuffer(
                    self.edges._bin_buffer,
                    dtype=self.edges._index.dtype,
                    count=end_block - start_block,
                    offset=start_block * self.edges._index.dtype().itemsize,
                )
                print(len(edges_buffer))
                assert len(edges_buffer) == head_entities_lens.sum()
                chunk_num_edges_per_head_entity = _count_num_edges_per_head_entity(
                    len(head_entities),
                    head_entities_pos,
                    head_entities_lens,
                    edges_buffer,
                    self.subsampling_cap,
                    10,
                )
                self.num_edges_per_head_entity[head_entities] = chunk_num_edges_per_head_entity
                start_block = end_block

            #     _, num_entity_pairs = np.unique(self.edges[head_entity].reshape(-1, self.EDGE_SIZE)[:, 0], return_counts=True)
            #     self.num_edges_per_head_entity[head_entity] = np.minimum(self.subsampling_cap,  num_entity_pairs).sum()
            # logger.info(
            #     'subsample graph by entity pairs: constructed num_edges_per_head_entity array in %d seconds. Total number of edges: %d' % (
            #         time.time() - start_time,
            #         self.num_edges_per_head_entity.sum(),
            #     ))

            print(time.time() - start_time)

        # loop over source entity
            # look over edges[source_entity]
            # collect all edges for a particular [target_entity]
            # sample subsample_cap of these edges


        index, dropped_sentences = 0, 0
        for edge_index in range(len(graph.index_to_sentences)):
            if graph.index_to_sentences.sizes[edge_index] == 1:
                sentence_id = graph.index_to_sentences[edge_index]
                if annotated_text_dataset.sizes[sentence_id[0]] < max_positions:
                    sentence_ids = sentence_id.numpy()
                else:
                    dropped_sentences += 1
                    continue
            else:
                sentence_lens = annotated_text_dataset.sizes[graph.index_to_sentences[edge_index]]
                sentence_ids = graph.index_to_sentences[edge_index][sentence_lens < max_positions].numpy()
                if len(sentence_ids) > subsample_cap:
                    sentence_ids = np.random.choice(sentence_ids, subsample_cap, replace=False)
                dropped_sentences += graph.index_to_sentences.sizes[edge_index] - len(sentence_ids)

            mtb_triplets[index:index + len(sentence_ids), 0] = sentence_ids
            mtb_triplets[index:index + len(sentence_ids), 1] = graph.index_to_entity_pair[edge_index][0]
            mtb_triplets[index:index + len(sentence_ids), 2] = graph.index_to_entity_pair[edge_index][1]
            index += len(sentence_ids)

        mtb_triplets = mtb_triplets[:index]

        dataset = MTBTripletsDataset(annotated_text_dataset, mtb_triplets)
        logger.info('subsample_graph_by_entity_pairs: generated %d examples (dropped %d edge-sentence pairs) in %d seconds' % (
            total_size,
            dropped_sentences,
            time.time() - start_time,
        ))
        return dataset