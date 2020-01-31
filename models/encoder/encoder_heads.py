from torch import nn
import torch

class BoW(nn.Module):

    def __init__(self, args):
        super().__init__()
        
    def forward(self, x):
        return torch.mean(x, dim=-2)

encoder_head_dict = {
    'bag_of_words': BoW
}

