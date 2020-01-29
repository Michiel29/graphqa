import numpy as np
from datasets import RelInfDataset

class TripletDataset(RelInfDataset):

    def __init__(self, text_data, annotation_data, k_negative, n_entities):
        self.text_data = text_data
        self.annotation_data = annotation_data
        self.k_negative = k_negative
        self.n_entities = n_entities


    def collater(self, instances):
    
        batch = {
            'mention': [],
            'head': [],
            'tail': [],
            'target': np.zeros(len(instances))
        }

        for instance in instances:

            """Perform Masking"""
            mention_samples, ent_samples = self.sample_entities(instance, self.k_negative)

            """Retrieve Subgraph"""
            for head, tail in ent_samples:
                batch['head'].append(head)
                batch['tail'].append(tail)
                
            batch['mentions'].append(mention_samples)


        return batch