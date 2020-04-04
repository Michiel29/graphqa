import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMult(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, mention, head, tail):
        return torch.bmm(head * tail, mention.unsqueeze(-1)).squeeze(-1)


class DistMultEntityOnly(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, mention, head, tail):
        return torch.dot(head, tail)


class RotatE(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, mention, head, tail):
        return ((mention.unsqueeze(-2) * head - tail) ** 2).sum(dim=-1)


class ConcatDot(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, mention, head, tail):
        head_tail_emb = torch.cat([head, tail], axis=-1)
        return torch.bmm(head_tail_emb, mention.unsqueeze(-1)).squeeze(-1)