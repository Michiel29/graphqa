from fairseq.data import MaskTokensDataset


class CustomMaskTokensDataset(MaskTokensDataset):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.epoch = None

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
        self.epoch = epoch