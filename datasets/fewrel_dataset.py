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
        text_data,
        annotation_data,
        relation_data,
        dictionary,
        mask_type,
        n_way,
        n_shot,
        dataset_size,
        shift_annotations,
        seed,
    ):
        self.text_data = text_data
        self.annotation_data = annotation_data
        self.relation_data = relation_data
        self.dictionary = dictionary
        self.dataset = AnnotatedTextDataset(
            text_data=self.text_data,
            annotation_data=self.annotation_data,
            dictionary=self.dictionary,
            shift_annotations=shift_annotations,
            assign_head_tail_randomly=False,
            mask_type=mask_type,
            seed=seed,
        )
        self.n_way = n_way
        self.n_shot = n_shot
        self.dataset_size = dataset_size
        self.seed = seed
        self.epoch = 0

        self.relation_index = defaultdict(list)
        for idx in range(len(self.relation_data)):
            self.relation_index[self.relation_data[idx].item()].append(idx)

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
                positive_mention_idxs = rd.choice(self.relation_index[positive_relation], size=self.n_shot + 1, replace=False)

                goal_mention_idx = positive_mention_idxs[0]
                exemplars += list(positive_mention_idxs[1:])

                # Sample n_shot exemplar sentences for other classes
                for rel in negative_relations:
                    rel_examplar_idxs = rd.choice(self.relation_index[rel], size=self.n_shot, replace=False)
                    exemplars += list(rel_examplar_idxs)

                all_ids = [goal_mention_idx] + [idx for idx in exemplars]
                total_tokens = sum([self.dataset.num_tokens(idx) for idx in all_ids])

                # Generate list of instances, each corresponding to a dict with id of goal mention, list of exemplars and the total nr of tokens in goal mention and exemplar sentences
                self.data.append({
                    'mention_id': goal_mention_idx,
                    'exemplars': exemplars,
                    'size': total_tokens,
                })

        self.sizes = np.array([instance['size'] for instance in self.data])

    def __getitem__(self, index):
        id_dict = self.data[index]
        return {
            'mention': self.dataset[id_dict['mention_id']]['mention'],
            'exemplars': [
                self.dataset[mention_id]['mention']
                for mention_id in id_dict['exemplars']
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

        mention = []
        exemplars = []
        ntokens, nsentences = 0, 0

        for instance in instances:
            mention.append(instance['mention'])
            exemplars += instance['exemplars']
            ntokens += sum([len(s) for s in instance['exemplars']])
            nsentences += 1 + len(instance['exemplars'])

        padded_mention = pad_sequence(mention, batch_first=True, padding_value=self.dictionary.pad())
        padded_exemplars = pad_sequence(exemplars, batch_first=True, padding_value=self.dictionary.pad())

        return {
            'mention': padded_mention,
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
