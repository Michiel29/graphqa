from torch import nn
import torch

class BoW(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.head_idx = dictionary.head()
        self.tail_idx = dictionary.tail()

    def forward(self, x, src_tokens):
        # x: [batch_size, length, enc_dim]
        mask = ((src_tokens != self.padding_idx) & (src_tokens != self.head_idx) & (src_tokens != self.tail_idx)).unsqueeze(-1) # [batch_size, length, enc_dim]
        masked_emb = x * mask
        mask_sum = mask.sum(dim=-2) # [batch_size, enc_dim]
        avg_emb = torch.sum(masked_emb, dim=-2) / mask_sum

        return avg_emb

class HeadTailConcat(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.head_idx = dictionary.head()
        self.tail_idx = dictionary.tail()

    def forward(self, x, src_tokens):
        # x: [batch_size, length, enc_dim]

        head_mask = (src_tokens == self.head_idx) # [batch_size, length]
        tail_mask = (src_tokens == self.tail_idx) # [batch_size, length]

        head_sum = torch.bmm(head_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]

        head_tail_concat = torch.cat((head_sum, tail_sum), dim=-1) # [batch_size, 2 * enc_dim]

        return head_tail_concat

encoder_head_dict = {
    'bag_of_words': BoW,
    'head_tail_concat': HeadTailConcat
}
