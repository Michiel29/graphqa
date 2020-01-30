import numpy as np
from datasets import RelInfDataset

class TripletDataset(RelInfDataset):

    def __init__(self, text_data, annotation_data, k_negative, n_entities, ent_tokens, ent_un_token):
        super().__init__(text_data, annotation_data, k_negative, n_entities, ent_tokens, ent_un_token)


    def collater(self, instances):
    
        batch = {
            'mention': [],
            'head': [],
            'tail': [],
            'target': np.zeros(len(instances))
        }

        for instance in instances:

            """Perform Masking"""
            mention, ent_samples = self.sample_entities(instance)

            """Retrieve Subgraph"""
            for head, tail in ent_samples:
                batch['head'].append(head)
                batch['tail'].append(tail)
                
            batch['mention'].append(mention)


        return batch