from collections import defaultdict
import logging
import numpy as np
import numpy.random as rd
import torch

from torch.nn.utils.rnn import pad_sequence

from fairseq.data import FairseqDataset
from fairseq.data.data_utils import numpy_seed
from datasets import AnnotatedTextDataset

logger = logging.getLogger(__name__)


class FewRelDataset(FairseqDataset):

    def __init__(
        self,
        annotation_text_dataset,
        relation_dataset,
        dictionary,
        mask_type,
        n_way,
        n_shot,
        dataset_size,
        seed,
    ):
        self.annotation_text_dataset = annotation_text_dataset
        self.relation_dataset = relation_dataset
        self.dictionary = dictionary
        self.n_way = n_way
        self.n_shot = n_shot
        self.dataset_size = dataset_size
        self.seed = seed
        self.epoch = 0

        self.relation_index = defaultdict(list)
        for idx in range(len(self.relation_dataset)):
            self.relation_index[self.relation_dataset[idx].item()].append(idx)

        self.data = []

        logger.info('creating %d examples with seed %d and epoch %d' % (dataset_size, self.seed, self.epoch))

        with numpy_seed(self.seed, self.epoch):
            for _ in range(self.dataset_size):

                exemplars = []

                # Sample n_way relation types and choose the first one as correct label
                sample_relations = rd.choice(list(self.relation_index.keys()), size=self.n_way, replace=False)
                positive_relation = sample_relations[0]
                negative_relations = sample_relations[1:]

                # Sample goal sentence + n_shot exemplar sentences for correct class
                positive_text_idxs = rd.choice(self.relation_index[positive_relation], size=self.n_shot + 1, replace=False)

                goal_text_idx = positive_text_idxs[0]
                exemplars += list(positive_text_idxs[1:])

                # Sample n_shot exemplar sentences for other classes
                for rel in negative_relations:
                    rel_examplar_idxs = rd.choice(self.relation_index[rel], size=self.n_shot, replace=False)
                    exemplars += list(rel_examplar_idxs)

                all_ids = [goal_text_idx] + [idx for idx in exemplars]
                total_tokens = sum([self.annotation_text_dataset.num_tokens(idx) for idx in all_ids])

                # Generate list of instances, each corresponding to a dict with id of goal text, list of exemplars and the total nr of tokens in goal text and exemplar sentences
                self.data.append({
                    'text_id': goal_text_idx,
                    'exemplars': exemplars,
                    'size': total_tokens,
                })

        self.sizes = np.array([instance['size'] for instance in self.data])

    def __getitem__(self, index):
        id_dict = self.data[index]
        return {
            'text': self.annotation_text_dataset[id_dict['text_id']]['text'],
            'exemplars': [
                self.annotation_text_dataset[text_id]['text']
                for text_id in id_dict['exemplars']
            ],
        }

    def __len__(self):
        return len(self.data)

    def num_tokens(self, index):
        return self.data[index]['size']

    def size(self, index):
        return self.data[index]['size']

    def ordered_indices(self):
        """Sorts by sentence length, randomly shuffled within sentences of same length"""
        return np.lexsort([
            np.random.permutation(len(self)),
            self.sizes,
        ])

    def collater(self, instances):
        batch_size = len(instances)

        if batch_size == 0:
            return None

        text = []
        exemplars = []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            exemplars += instance['exemplars']
            ntokens += len(instance['text']) + sum([len(s) for s in instance['exemplars']])
            nsentences += 1 + len(instance['exemplars'])

        padded_mention = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())
        padded_exemplars = pad_sequence(exemplars, batch_first=True, padding_value=self.dictionary.pad())

        return {
            'text': padded_mention,
            'exemplars': padded_exemplars,
            'target': torch.zeros(len(instances), dtype=torch.long),
            'batch_size': len(instances),
            'ntokens': ntokens,
            'nsentences': nsentences,
        }

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
