from fairseq.data import FairseqDataset

class RelInfDataset(FairseqDataset):

    def collater(self, instances):
        raise NotImplementedError

    def sample_entities(self, sample, k_negative):
        raise NotImplementedError