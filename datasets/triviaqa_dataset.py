from collections import defaultdict
import numpy as np
import numpy.random as rd
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import FairseqDataset

from utils.data_utils import numpy_seed

from datasets import AnnotatedText


class TriviaQADataset(FairseqDataset):

    def __init__(
        self,
        questions,
        answers,
        annotations,
    ):
        self.questions = questions
        self.answers = answers
        self.annotations = annotations
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):

        item = {
            'question': self.questions[index],
            'answer': self.answers.array[index],
            'annotation': self.annotations[index],

        }

        return item

    def __len__(self):
        return len(self.questions)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @property
    def sizes(self):
        return self.questions.sizes

    def ordered_indices(self):
        return np.argsort([10 * (np.random.random(len(self.sizes)) - 0.5) + self.sizes])[0]

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
