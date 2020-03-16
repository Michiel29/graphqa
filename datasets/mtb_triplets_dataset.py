import logging
import numpy as np
import time

from fairseq.data import BaseWrapperDataset

from . import FilteredDataset

logger = logging.getLogger(__name__)


class MTBTripletsDataset(BaseWrapperDataset):

    def __init__(
        self,
        dataset,
        mtb_triplets,
    ):
        self.mtb_triplets = mtb_triplets
        self.dataset = FilteredDataset(dataset, mtb_triplets[:, 0])

    def __getitem__(self, index):
        head = self.mtb_triplets[index][1]
        tail = self.mtb_triplets[index][2]
        return self.dataset.__getitem__(index, head_entity=head, tail_entity=tail)


def subsample_graph_by_entity_pairs(annotated_text_dataset, graph, subsample_cap, max_positions):
    start_time = time.time()
    total_size = np.minimum(subsample_cap, graph.index_to_sentences.sizes).sum()
    mtb_triplets = np.zeros((total_size, 3), dtype=np.int32)
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