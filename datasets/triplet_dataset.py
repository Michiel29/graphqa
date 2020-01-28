from datasets import RelInfDataset

class TripletDataset(RelInfDataset):

    def __init__(self, k_negative):
        self.k_negative = k_negative

    def collater(self, instances):
    
        batch = {
            'mention': [],
            'head': [],
            'tail': [],
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
            
