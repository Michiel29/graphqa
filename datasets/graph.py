from fairseq.data import FairseqDataset


class GraphDataset(FairseqDataset):

    def __init__(
        self,
        entity_neighbors,
        entity_edges,
        index_to_entity_pair,
        index_text_count,
        index_to_sentences,
    ):
        self.entity_neighbors = entity_neighbors
        self.entity_edges = entity_edges
        self.index_to_entity_pair = index_to_entity_pair
        self.index_text_count = index_text_count
        self.index_to_sentences = index_to_sentences

        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        neighbors = self.entity_neighbors[index]
        edges = self.entity_edges[index]
        return {
            'neighbors': neighbors,
            'edges': edges,
        }

    def __len__(self):
        return len(self.entity_neighbors)
