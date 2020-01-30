import torch
import torch.nn as nn
import torch.nn.functional as F

class DistMult(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.mode = getattr(args, 'dist_mode', '')

    def forward(self, mention, head, tail):

        if self.mode == 'head-batch':
            score = head * (mention * tail)
        else:
            score = (head * mention) * tail

        score = score.sum(dim = -1)

        return score
