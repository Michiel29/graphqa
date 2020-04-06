import numpy as np


class MMapNumpyArray(object):
    def __init__(self, path):
        self.path = path
        self.array = np.load(self.path, mmap_mode='r')

    def __getstate__(self):
        return self.path

    def __setstate__(self, path):
        self.__init__(path)
