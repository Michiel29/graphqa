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
