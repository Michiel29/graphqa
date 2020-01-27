from fairseq.data import FairseqDataset

class RelInfDataset(FairseqDataset):

    def __init__(self, args):
        self.args = args

    def collater(self, instances):

        batch = {
            'mentions': [],
            'subgraphs': [],
            'targets': []
        }

        for instance in instances:

            """Perform Masking"""
            mention_samples, ent_samples, targets = self.sample_entities(instance, self.args.k_negative)
            subgraphs = []

            """Retrieve Subgraph"""
            for ent1, ent2 in ent_samples:
                subgraph = self.retrieve_subgraph(ent1, ent2)
                subgraphs.append(subgraph)
            
            batch['mentions'].append(mention_samples)
            batch['subgraphs'].append(subgraphs)
            batch['targets'].append(targets)

        return batch

    def retrieve_subgraph(self, ent1, ent2):
        raise NotImplementedError

    def sample_entities(self, sample, k_negative):
        raise NotImplementedError