from pudb import set_trace
from torch import nn
import torch

class BoW(nn.Module):

    def __init__(self, args):
        super().__init__()

    def forward(self, x, padding_mask=None):
        # x: [batch_size, length, enc_dim]
        if padding_mask is not None:
            return x.sum(dim=1) / padding_mask.sum(dim=1, keepdims=True) # [batch_size, enc_dim]
        else:
            return torch.mean(x, dim=1) # [batch_size, enc_dim]

class HeadTailConcat(nn.Module):

    def __init__(self, args):
        super().__init__()

    def forward(self, x, head_tail_mask):
        # x: [batch_size, length, enc_dim]

        batch_size = x.shape[0]
        enc_dim = x.shape[2]

        head_tail_mask = head_tail_mask.unsqueeze(-1).expand(-1, -1, enc_dim)
        return torch.masked_select(x, head_tail_mask).reshape(batch_size, 2 * enc_dim)


encoder_head_dict = {
    'bag_of_words': BoW,
    'head_tail_concat': HeadTailConcat
}
