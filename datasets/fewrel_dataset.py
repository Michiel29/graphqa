from collections import defaultdict

import numpy as np
import numpy.random as rd
import torch

from torch.nn.utils.rnn import pad_sequence

from fairseq.data import FairseqDataset

from datasets import AnnotatedTextDataset


class FewRelDataset(FairseqDataset):

    def __init__(
        self,
        text_data,
        annotation_data,
        relation_data,
        dictionary,
        n_way,
        n_shot,
        dataset_size,
        shift_annotations,
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
        )

        self.n_way = n_way
        self.n_shot = n_shot
        self.dataset_size = dataset_size

        self.relation_index = defaultdict(list)
        for idx in range(len(self.relation_data)):
            self.relation_index[self.relation_data[idx].item()].append(idx)

        self.data = []

        # Exact validation set depends on the sampling being used. Thus, we
        # 1. Save the current random generation state
        # 2. Set random seed to be a paricular constant
        # 3. We restore random state after the data generation has finished.
        old_rd_state = rd.get_state()
        rd.seed(31415)

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

        # Restore the random state back
        rd.set_state(old_rd_state)

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

        mention = []
        exemplars = []

        for instance in instances:
            mention.append(instance['mention'])
            exemplars += instance['exemplars']

        padded_mention = pad_sequence(mention, batch_first=True, padding_value=self.dictionary.pad())
        padded_exemplars = pad_sequence(exemplars, batch_first=True, padding_value=self.dictionary.pad())

        batch = {}

        batch['mention'] = padded_mention
        batch['exemplars'] = padded_exemplars
        batch['target'] = torch.zeros(len(instances), dtype=torch.long)
        batch['batch_size'] = len(instances)
        batch['ntokens'] = sum(len(m) for m in mention)
        batch['nsentences'] = len(padded_exemplars)

        return batch

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
