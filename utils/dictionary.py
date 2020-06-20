from fairseq.data import Dictionary


class CustomDictionary(Dictionary):
    """Dictionary with entity tokens"""

    head_token_df = '<head>'
    tail_token_df = '<tail>'

    blank_token_df = '<blank>'
    e1_start_token_df = '<s_head>'
    e1_end_token_df = '<e_head>'
    e2_start_token_df = '<s_tail>'
    e2_end_token_df = '<e_tail>'

    blank_head_other_df = '<blank_head_other>'
    blank_tail_other_df = '<blank_tail_other>'

    pad_df="<pad>"
    eos_df="</s>"
    unk_df="<unk>"
    bos_df="<s>"

    def __init__(self,
        head_token=head_token_df,
        tail_token=tail_token_df,
        blank_token=blank_token_df,
        e1_start_token=e1_start_token_df,
        e1_end_token=e1_end_token_df,
        e2_start_token=e2_start_token_df,
        e2_end_token=e2_end_token_df,
        pad=pad_df,
        eos=eos_df,
        unk=unk_df,
        bos=bos_df,
        blank_head_other_token=blank_head_other_df,
        blank_tail_other_token=blank_tail_other_df,
    ):
        super().__init__(pad, eos, unk, bos)

        self.head_token = head_token
        self.tail_token = tail_token

        self.blank_token = blank_token
        self.e1_start_token = e1_start_token
        self.e1_end_token = e1_end_token
        self.e2_start_token = e2_start_token
        self.e2_end_token = e2_end_token

        self.blank_head_other_token = blank_head_other_token
        self.blank_tail_other_token = blank_tail_other_token

    def add_from_file(self, f):
        Dictionary.add_from_file(self, f)

        self.head_index = self.add_symbol(self.head_token)
        self.tail_index = self.add_symbol(self.tail_token)

        self.blank_index = self.add_symbol(self.blank_token)
        self.e1_start_index = self.add_symbol(self.e1_start_token)
        self.e1_end_index = self.add_symbol(self.e1_end_token)
        self.e2_start_index = self.add_symbol(self.e2_start_token)
        self.e2_end_index = self.add_symbol(self.e2_end_token)

        self.blank_head_other_index = self.add_symbol(self.blank_head_other_token)
        self.blank_tail_other_index = self.add_symbol(self.blank_tail_other_token)

        self.mask_index = self.index('<mask>')

    def head(self):
        return self.head_index

    def tail(self):
        return self.tail_index

    def blank(self):
        return self.blank_index

    def e1_start(self):
        return self.e1_start_index

    def e1_end(self):
        return self.e1_end_index

    def e2_start(self):
        return self.e2_start_index

    def e2_end(self):
        return self.e2_end_index

    def mask(self):
        return self.mask_index

    def blank_head_other(self):
        return self.blank_head_other_index

    def blank_tail_other(self):
        return self.blank_tail_other_index

    def special_tokens(self):
        return [
            self.head(),
            self.tail(),
            self.blank(),
            self.e1_start(),
            self.e1_end(),
            self.e2_start(),
            self.e2_end(),
            self.pad(),
            self.mask(),
            self.blank_head_other(),
            self.blank_tail_other(),
        ]


class EntityDictionary(Dictionary):
    """Dictionary with no special tokens"""
    def __init__(self):
        self.symbols = []
        self.count = []
        self.indices = {}

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:
        <symbol0> <count0>
        <symbol1> <count1>
        """
        d = cls()
        d.add_from_file(f)
        return d