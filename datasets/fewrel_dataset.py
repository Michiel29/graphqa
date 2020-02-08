from collections import defaultdict

import numpy as np
import numpy.random as rd
import torch

from torch.nn.utils.rnn import pad_sequence


from fairseq.data import FairseqDataset

class FewRelDataset(FairseqDataset):

    def __init__(self, text_data, annotation_data, relation_data, 
    dictionary, n_way, n_shot, dataset_size):
        self.text_data = text_data
        self.annotation_data = annotation_data
        self.relation_data = relation_data
        self.dictionary = dictionary
        self.n_way = n_way
        self.n_shot = n_shot


        self.processed_mentions = []

        self.relation_index = defaultdict(list)

        bos_offset = int(hasattr(self.text_data, 'token'))

        for idx in range(len(relation_data)):
            self.relation_index[relation_data[idx].item()].append(idx)

            annotation = annotation_data[idx].split(3)
            mention = text_data[idx]
            ent_tokens = [self.dictionary.head(), self.dictionary.tail()]
            for entity_annotation in annotation:
                ent_slice = slice(entity_annotation[0] + bos_offset, entity_annotation[1] + bos_offset)
                mention[ent_slice] = -1
                mention[entity_annotation[0] + bos_offset] = ent_tokens[entity_annotation[2]]
            
            self.processed_mentions.append(mention[mention!=-1])


        self.data = []

        for _ in range(dataset_size):
            
            exemplars = []

            sample_relations = rd.choice(list(self.relation_index.keys()), size=self.n_way, replace=False)
            positive_relation = sample_relations[0]
            negative_relations = sample_relations[1:]

            positive_mention_idxs = rd.choice(self.relation_index[positive_relation], size=self.n_shot + 1, replace=False)
            
            goal_mention_idx = positive_mention_idxs[0]
            
            exemplars += list(positive_mention_idxs[1:])

            for rel in negative_relations:

                rel_examplar_idxs = rd.choice(self.relation_index[rel], size=self.n_shot, replace=False)
                exemplars += list(rel_examplar_idxs)

            all_ids = [goal_mention_idx] + [idx for idx in exemplars]
            total_tokens = sum([len(self.processed_mentions[idx]) for idx in all_ids])

            id_dict = { 
            'mention_id': goal_mention_idx,
            'exemplars': exemplars,
            'size': total_tokens
            }   

            self.data.append(id_dict)

        self.sizes = np.array([instance['size'] for instance in self.data])

    def __getitem__(self, index):
        id_dict = self.data[index]
        
        item_dict = {}
        item_dict['mention'] = self.processed_mentions[id_dict['mention_id']]
        item_dict['exemplars'] = [self.processed_mentions[mention_id] for mention_id in id_dict['exemplars']]

        return item_dict

    def __len__(self):
        return len(self.data)

    def num_tokens(self, index):
        return self.data[index]['size']

    def size(self, index):
        return self.data[index]['size']

    def ordered_indices(self):
        """Sorts by sentence length, randomly shuffled within sentences of """
        order = np.arange(len(self))
        np.random.shuffle(order)
        order = [order]
        order.append(self.sizes)        
        indices = np.lexsort(order)

        return indices

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

        return batch

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False 

