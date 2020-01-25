from fairseq.data import FairseqDataset

class RelInfDataset(FairseqDataset):
    def collater(self, samples):
        return NotImplemented