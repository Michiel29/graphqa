import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMult(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, mention, head, tail):
        score = head * (mention * tail)
        score = score.sum(dim = -1)
        return score


class DistMultEntityOnly(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, mention, head, tail):
        return torch.dot(head, tail)


class RotatE(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, mention, head, tail):
        score = head * mention - tail
        score = (score ** 2).sum(dim = -1)
        return score