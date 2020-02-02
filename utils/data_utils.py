from fairseq.data import Dictionary


class CustomDictionary(Dictionary):

    def __init__(self, head_token='<head>', tail_token = '<tail>', unk_ent_token='<unk_ent>'):
        super().__init__()
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
    def __init__(self):
        self.symbols = []
        self.count = []
        self.indices = {}




