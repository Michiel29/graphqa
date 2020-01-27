from datasets import RelInfDataset

class TripletDataset(RelInfDataset):
    def retrieve_subgraph(self, ent1, ent2):
        return (self.ent_dict[ent1], self.)
        raise NotImplementedError

    def sample_entities(self, sample, k_negative):
        raise NotImplementedError