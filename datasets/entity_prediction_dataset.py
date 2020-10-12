import logging
import math
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import FairseqDataset

from datasets import GraphDataset
from utils.data_utils import numpy_seed


logger = logging.getLogger(__name__)


class EntityPredictionDataset(FairseqDataset):

    def __init__(
        self,
        annotated_text,
        edges,
        dictionary,
        total_negatives,
        max_tokens,
        max_sentences,
        num_text_chunks,
        num_workers,
        seed,
    ):
        self.annotated_text = annotated_text
        self.edges = edges
        self.dictionary = dictionary

        self.total_negatives = total_negatives

        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.num_text_chunks = num_text_chunks
        self.seed = seed
        self.epoch = None

        self.num_workers = num_workers

    def set_epoch(self, epoch):
        self.epoch = epoch


    def __getitem__(self, index):
        with numpy_seed('EntityPredictionDataset', self.seed, self.epoch, index):
            entity = np.random.randint(len(self.edges))
            passage_idx = np.random.randint(len(self.edges[entity]))
            edge = self.edges[entity][passage_idx]
            start_pos, end_pos, start_block, end_block = edge[GraphDataset.HEAD_START_POS], edge[GraphDataset.HEAD_END_POS], edge[GraphDataset.START_BLOCK], edge[GraphDataset.END_BLOCK]
            passage, annotation_position = self.annotated_text.annotate_mention(entity, start_pos, end_pos, start_block, end_block)
            negatives = np.random.choice(len(self.edges), replace=False, size=self.total_negatives)
            candidates = np.concatenate(([entity], negatives))

        item = {
            'text': passage,
            'annotation': annotation_position,
            'candidates': candidates,
        }
        return item

    def __len__(self):
        return len(self.edges)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @property
    def sizes(self):
        return self.graph.sizes

    def ordered_indices(self):
        return self.graph.ordered_indices()

    def collater(self, instances):
        batch_size = len(instances)

        if batch_size == 0:
            return None

        text, annotation = [], []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            annotation.append(instance['annotation'])
            ntokens += len(instance['text'])
            nsentences += 1

        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())

        if len(annotation) > 0 and annotation[0] is not None:
            annotation = torch.LongTensor(annotation)
        else:
            annotation = None

        batch = {
            'text': padded_text,
            'target': torch.zeros(batch_size, dtype=torch.int64),
            'annotation': annotation,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }

        return batch
