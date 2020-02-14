from fairseq.data import Dictionary


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




