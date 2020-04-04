import logging
import numpy as np


logger = logging.getLogger(__name__)


class MMapNumpyArray(object):
    def __init__(self, path):
        self.path = path
        self.array = np.load(self.path, mmap_mode='r')
        logger.info('memory mapped array of shape %s from %s' % (
            str(self.array.shape),
            self.path,
        ))

    def __getstate__(self):
        return self.path

    def __setstate__(self, path):
        self.__init__(path)