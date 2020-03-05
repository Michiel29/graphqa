# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from fairseq.data import BaseWrapperDataset


class ShuffledDataset(BaseWrapperDataset):

    def __init__(self, dataset, sizes):
        super().__init__(dataset)
        self._sizes = sizes

    @property
    def sizes(self):
        return self._sizes

    def ordered_indices(self):
        return np.lexsort([
            np.random.permutation(len(self)),
            self.sizes,
        ])
