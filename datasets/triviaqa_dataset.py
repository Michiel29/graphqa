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
        dictionary,
        seed,
    ):
        self.questions = questions
        self.answers = answers
        self.annotations = annotations
        self.dictionary = dictionary
        self.seed = seed
        self.epoch = 0



    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed('triviaqa', self.seed, self.epoch, index):
            pass

        item = {
            # 'text': annot_item,
            # 'target': relation
        }

        return item

    def __len__(self):
        return len(self.annotation_text)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @property
    def sizes(self):
        return self.annotation_text.sizes

    def ordered_indices(self):
        return np.argsort([10 * (np.random.random(len(self.sizes)) - 0.5) + self.sizes])[0]

    def collater(self, instances):
        batch_size = len(instances)

        if batch_size == 0:
            return None

        text, target = [], []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            target.append(instance['target'])
            ntokens += len(instance['text'])
            nsentences += 1

        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())

        batch = {
            'text': padded_text,
            'target': torch.LongTensor(target),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }

        return batch

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
