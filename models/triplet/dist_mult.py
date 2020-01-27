import torch
import torch.nn as nn
import torch.nn.functional as F

class DistMult(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.mode = getattr(args, 'dist_mode', '')

    def forward(self, x_emb):

        head = x_emb['head']
        tail = x_emb['tail']
        relation = x_emb['goal']

        if self.mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = -1)

        return score