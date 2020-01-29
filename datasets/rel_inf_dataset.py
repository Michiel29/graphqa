import torch

import numpy as np
import numpy.random as rd

from fairseq.data import FairseqDataset

class RelInfDataset(FairseqDataset):

    def __init__(self, text_data, annotation_data, n_entities):
        self.text_data = text_data
        self.annotation_data = self.annotation_data
        self.n_entities = n_entities

    def __getitem__(self, index):
        item_dict = {
        'mention': self.text_data[index],
        'annotation': self.annotation_data
        }
        return item_dict

    def __len__(self):
        return len(self.text_data)

    def num_tokens(self, index):
        return self.text_data.sizes[index]

    def size(self, index):
        return self.text_data.sizes[index]

    def ordered_indices(self):
        order = [np.arange(len(self))]
        order.append(self.text_data.sizes)        
        indices = np.lexsort(order)

        return indices
    
    def sample_entities(self, instance, k_negative):

        # Need to sort out random seed properly

        mention = instance['mention']
        annotation = instance['annotation']

        entities = torch.split(annotation, 3)

        mention_entity_ids = rd.choice(len(entities), 2, replace=False)
        # check indexing here
        mention_entities = entities[mention_entity_ids]

        # for entity in mention_entities: 
        #     mention[]




        ent_samples = [] 
        mention_samples = []

        # entity_to_replace = rd.binomial(k_negative, 0.5)
        # replacement_entities = rd.randint(self.n_entities, size=k_negative)

        # heads = []
        tails = []

        

        # instance['mention']
        # instance['annotation']

        # mention_samples, ent_samples = self.sample_entities(instance, self.k_negative)

        return mention_samples, ent_samples

    def collater(self, instances):
        raise NotImplementedError        

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False        


