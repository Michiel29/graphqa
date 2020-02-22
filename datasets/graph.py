from fairseq.data import FairseqDataset

class GraphDataset(FairseqDataset):

    def __init__(
        self,
        entity_neighbors,
        entity_edges,
    ):
        self.entity_neighbors = entity_neighbors
        self.entity_edges = entity_edges

    def __getitem__(self, index):
        return {
            'neighbors': self.entity_neighbors[index],
            'edges': self.entity_edges[index]
        }

    def __len__(self):
        return len(self.entity_neighbors)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False