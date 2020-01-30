import torch

import numpy as np
import numpy.random as rd

from fairseq.data import FairseqDataset

class RelInfDataset(FairseqDataset):

    def __init__(self, text_data, annotation_data, k_negative, n_entities, dictionary):
        self.text_data = text_data
        self.annotation_data = annotation_data
        self.k_negative = k_negative
        self.n_entities = n_entities

        self.dictionary = dictionary

    def __getitem__(self, index):
        item_dict = {
        'mention': self.text_data[index],
        'annotation': self.annotation_data[index]
        }
        return item_dict

    def __len__(self):
        return len(self.text_data)

    def num_tokens(self, index):
        return self.text_data.sizes[index]

    def size(self, index):
        return self.text_data.sizes[index]

    def ordered_indices(self):
        """Sorts by sentence length, randomly shuffled within sentences of """
        order = np.arange(len(self))
        np.random.shuffle(order)
        order = [order]
        order.append(self.text_data.sizes)        
        indices = np.lexsort(order)

        return indices
    
    def sample_entities(self, instance):

        # Need to sort out random seed properly

        mention = instance['mention']
        annotation = instance['annotation']

        entities = torch.split(annotation, 3)

        mention_entity_ids = rd.choice(len(entities), 2, replace=False)
        mention_entities = [entities[idx] for idx in mention_entity_ids]

        ent_tokens = [self.dictionary.head(), self.dictionary.tail()]

        # remove entity tokens from mention
        for i, entity in enumerate(mention_entities): 
            entity_slice = slice(entity[0], entity[1])
            mention[entity_slice] = -1

            # replace with appropriate head/tail ent token
            mention[entity[0]] = ent_tokens[i]

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


