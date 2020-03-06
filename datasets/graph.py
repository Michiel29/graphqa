from fairseq.data import FairseqDataset


class GraphDataset(FairseqDataset):

    def __init__(
        self,
        entity_neighbors,
        entity_edges
    ):
        self.entity_neighbors = entity_neighbors
        self.entity_edges = entity_edges
        self.edges_to_keep = None
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        # if self.edges_to_keep is None:
        neighbors = self.entity_neighbors[index]
        edges = self.entity_edges[index]
        # else:
        #     neighbors
        #     edges_to_keep
        #     for i in

        return {
            'neighbors': neighbors,
            'edges': edges,
        }


    def __len__(self):
        return len(self.entity_neighbors)

    def set_edges_to_keep(self, edges_to_keep):
        self.edges_to_keep = frozenset(edges_to_keep)
