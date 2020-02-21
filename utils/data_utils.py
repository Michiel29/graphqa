from fairseq.data import Dictionary
from collections import defaultdict
from itertools import combinations

class CustomDictionary(Dictionary):
    """Dictionary with entity tokens"""

    head_token_df = '<head>'
    tail_token_df = '<tail>'
    unk_ent_token_df = '<unk_ent>'
    pad_df="<pad>"
    eos_df="</s>"
    unk_df="<unk>"
    bos_df="<s>"

    def __init__(self, head_token=head_token_df, tail_token = tail_token_df, unk_ent_token=unk_ent_token_df,
    pad=pad_df, eos=eos_df, unk=unk_df, bos=bos_df):
        super().__init__(pad, eos, unk, bos)
        self.head_token = head_token
        self.tail_token = tail_token
        self.unk_ent_token = unk_ent_token

    def add_from_file(self, f):
        Dictionary.add_from_file(self, f)
        self.head_index = self.add_symbol(self.head_token)
        self.tail_index = self.add_symbol(self.tail_token)
        self.unk_ent_index = self.add_symbol(self.unk_ent_token)

    def head(self):
        return self.head_index

    def tail(self):
        return self.tail_index

    def unk_ent(self):
        return self.unk_ent_index

class EntityDictionary(Dictionary):
    """Dictionary with no special tokens"""
    def __init__(self):
        self.symbols = []
        self.count = []
        self.indices = {}

class Graph():

    def __init__(self):
        self.edge_dict = None
        self.entity_neighbors = None

    def construct_graph(self, annotation_data, n_entities):

        self.entity_neighbors = [defaultdict(int) for entity in range(n_entities)]
        self.edge_dict = defaultdict(list)

        for sentence_idx in range(len(annotation_data)):
            entity_ids = annotation_data[sentence_idx].reshape(-1, 3)[:, -1].numpy()

            for a, b in combinations(entity_ids, 2):
                self.entity_neighbors[a][b] += 1
                self.entity_neighbors[b][a] += 1

                self.edge_dict[frozenset({a, b})].append(sentence_idx)



