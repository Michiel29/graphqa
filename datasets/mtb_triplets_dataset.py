import numpy as np
from fairseq.data import BaseWrapperDataset

from . import FilteredDataset


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
        return {
            'text': self.dataset.__getitem__(index, head_entity=head, tail_entity=tail)['text'],
            'head': head,
            'tail': tail,
        }