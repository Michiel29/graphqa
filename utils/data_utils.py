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

class MTBDictionary(Dictionary):
    """Dictionary with entity tokens"""

    e1_start_token_df = '<E1>'
    e1_end_token_df = '<\E1>'
    e2_start_token_df = '<E2>'
    e2_end_token_df = '<\E2>'
    blank_token_df = '<BLANK>'
    pad_df="<pad>"
    eos_df="</s>"
    unk_df="<unk>"
    bos_df="<s>"

    def __init__(self, e1_start_token=e1_start_token_df, 
                        e1_end_token=e1_end_token_df, 
                        e2_start_token=e2_start_token_df, 
                        e2_end_token=e2_end_token_df, 
                        blank_token=blank_token_df,
                        pad=pad_df, eos=eos_df, unk=unk_df, bos=bos_df):
        super().__init__(pad, eos, unk, bos)
        self.e1_start_token = e1_start_token
        self.e1_end_token = e1_end_token
        self.e2_start_token = e2_start_token
        self.e2_end_token = e2_end_token
        self.blank_token = blank_token

    def add_from_file(self, f):
        Dictionary.add_from_file(self, f)
        self.e1_start_index = self.add_symbol(self.e1_start_token)
        self.e1_end_index = self.add_symbol(self.e1_end_token)
        self.e2_start_index = self.add_symbol(self.e2_start_token)
        self.e2_end_index = self.add_symbol(self.e2_end_token)
        self.blank_index = self.add_symbol(self.blank_token)

    def e1_start(self):
        return self.e1_start_index

    def e1_end(self):
        return self.e1_end_index

    def e2_start(self):
        return self.e2_start_index

    def e2_end(self):
        return self.e2_end_index

    def blank(self):
        return self.blank_index
    '''
    def unk_ent(self):
        return self.unk_ent_index
    '''
class EntityDictionary(Dictionary):
    """Dictionary with no special tokens"""
    def __init__(self):
        self.symbols = []
        self.count = []
        self.indices = {}




