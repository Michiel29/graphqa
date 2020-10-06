from collections import defaultdict
import numpy as np
import numpy.random as rd
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import data_utils, FairseqDataset

from datasets import AnnotatedText


class KBP37Dataset(FairseqDataset):

    def __init__(
        self,
        annotation_text,
        relation_dataset,
        dictionary,
        seed,
    ):
        self.annotation_text = annotation_text
        self.relation_dataset = relation_dataset
        self.dictionary = dictionary
        self.seed = seed
        self.epoch = 0

        self.relation_index = defaultdict(list)
        for idx in range(len(self.relation_dataset)):
            self.relation_index[self.relation_dataset[idx].item()].append(idx)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed(271828, self.seed, self.epoch, index):
            annot_item, annotation = self.annotation_text.annotate_sentence(index, head_entity=1, tail_entity=2)
            relation = self.relation_dataset[index]

        item = {
            'text': annot_item,
            'annotation': annotation,
            'target': relation
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

        text, target, annotation = [], [], []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            target.append(instance['target'])
            annotation.append(instance['annotation'])
            ntokens += len(instance['text'])
            nsentences += 1

        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())

        batch = {
            'text': padded_text,
            'target': torch.LongTensor(target),
            'annotation': torch.LongTensor(annotation),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }

        return batch

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
