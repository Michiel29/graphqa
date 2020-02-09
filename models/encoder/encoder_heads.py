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


encoder_head_dict = {
    'bag_of_words': BoW,
}
