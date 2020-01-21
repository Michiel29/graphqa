import torch
import torch.nn as nn
import torch.nn.functional as F

class DistMult(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x_emb, mode='single'):

        head = x_emb['head']
        tail = x_emb['tail']
        relation = x_emb['goal']

        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = -1)

        return score

triplet_dict = {'distmult': DistMult}

