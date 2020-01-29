from fairseq.data import FairseqDataset

class FewRelDataset(FairseqDataset):

    def __init__(self, text_data, annotation_data, n_way, n_shot):
        self.text_data = text_data
        self.annotation_data = self.annotation_data
        self.n_way = n_way
        self.n_shot = n_shot

    def __getitem__(self, index):
        item_dict = { 
        'goal_mention': self.text_data[index],
        'candidate_mentions': [],
        'annotation': self.annotation_data
        }   
        return item_dict

    def __len__(self):
        return len(self.text_data)

    def num_tokens(self, index):
        return self.text_data.sizes[index]

    def size(self, index):
        return self.text_data.sizes[index]


    def collater(self, instances):
        raise NotImplementedError    

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False 

