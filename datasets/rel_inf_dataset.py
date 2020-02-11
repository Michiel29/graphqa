import torch

import numpy as np
import numpy.random as rd

from fairseq.data import FairseqDataset

class RelInfDataset(FairseqDataset):

    def __init__(self, text_data, annotation_data, k_negative, n_entities, dictionary, n_examples=None):
        self.text_data = text_data
        self.annotation_data = annotation_data
        self.k_negative = k_negative
        self.n_entities = n_entities

        self.dictionary = dictionary

        self.n_examples = n_examples
        if n_examples is not None:
            self.dataset_indices = range(n_examples)
        else:
            self.dataset_indices = None

    def __getitem__(self, index):
        if self.dataset_indices is None:
            item_dict = {
            'mention': self.text_data[index],
            'annotation': self.annotation_data[index]
            }
        else: 
            item_dict = {
            'mention': self.text_data[self.dataset_indices[index]],
            'annotation': self.annotation_data[self.dataset_indices[index]]
            }

        return item_dict

    def __len__(self):
        if self.dataset_indices is None:
            return len(self.text_data)
        else:
            return self.n_examples

    def num_tokens(self, index):
        if self.dataset_indices is None:
            return self.text_data.sizes[index]
        else: 
            return self.text_data.sizes[self.dataset_indices[index]]

    def size(self, index):
        if self.dataset_indices is None:
            return self.text_data.sizes[index]
        else: 
            return self.text_data.sizes[self.dataset_indices[index]]

    def ordered_indices(self):
        """Sorts by sentence length, randomly shuffled within sentences of """
        order = np.arange(len(self))
        np.random.shuffle(order)
        order = [order]

        if self.dataset_indices is None:
            order.append(self.text_data.sizes)
        else:
            order.append(self.text_data.sizes[self.dataset_indices])

        indices = np.lexsort(order)

        return indices
    
    def sample_entities(self, instance):

        # Need to sort out random seed properly

        mention = instance['mention']
        annotation = instance['annotation']

        entities = torch.split(annotation, 3)

        mention_entity_ids = rd.choice(len(entities), 2, replace=False)
        mention_entities = [entities[idx] for idx in mention_entity_ids]

        # offset annotations by 1 if bos token added
        bos_offset = int(hasattr(self.text_data, 'token'))

        ent_tokens = [self.dictionary.head(), self.dictionary.tail()]

        # remove entity tokens from mention
        for i, entity in enumerate(mention_entities): 
            entity_slice = slice(entity[0] + bos_offset, entity[1] + bos_offset)
            mention[entity_slice] = -1

            # replace with appropriate head/tail ent token
            mention[entity[0] + bos_offset] = ent_tokens[i]

        mention = mention[mention!=-1]        

        entity_to_replace = rd.binomial(self.k_negative, 0.5, size=self.k_negative)
        replacement_entities = rd.randint(self.n_entities, size=self.k_negative)

        head_ent = mention_entities[0][2].item()
        tail_ent = mention_entities[1][2].item()

        heads = [head_ent] + [head_ent if entity_to_replace[i]==1 else replacement_entities[i] for i in range(self.k_negative)] 
        tails = [tail_ent] + [tail_ent if entity_to_replace[i]==0 else replacement_entities[i] for i in range(self.k_negative)]

    
        return mention, heads, tails

    def collater(self, instances):
        raise NotImplementedError        

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False        


