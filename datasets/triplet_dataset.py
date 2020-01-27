from datasets import RelInfDataset

class TripletDataset(RelInfDataset):
    def retrieve_subgraph(self, ent1, ent2):
        raise NotImplementedError

    def sample_entities(self, sample, k_negative):
        raise NotImplementedError