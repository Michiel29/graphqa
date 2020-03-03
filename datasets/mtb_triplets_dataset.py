import numpy as np
from fairseq.data import FairseqDataset

class MTBTripletsDataset(FairseqDataset):

    def __init__(
        self,
        mtb_triplets,
    ):
        self.mtb_triplets = mtb_triplets
        self.sentence_array = np.array([x[0] for x in mtb_triplets])
    def __getitem__(self, index):
        return self.mtb_triplets[index]

    def __len__(self):
        return len(self.mtb_triplets)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
