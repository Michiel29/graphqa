import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMult(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.scaling = args.entity_dim ** -0.5

    def forward(self, mention, head, tail):
        return self.scaling * torch.bmm(head * tail, mention.unsqueeze(-1)).squeeze(-1)


class DistMultEntityOnly(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.scaling = args.entity_dim ** -0.5

    def forward(self, mention, head, tail):
        return self.scaling * torch.dot(head, tail)


class RotatE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.scaling = args.entity_dim ** -0.5

    def forward(self, mention, head, tail):
        return self.scaling * ((mention.unsqueeze(-2) * head - tail) ** 2).sum(dim=-1)


class ConcatDot(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.scaling = (2 * args.entity_dim) ** -0.5

    def forward(self, mention, head, tail):
        head_tail_emb = torch.cat([head, tail], axis=-1)
        return self.scaling * torch.bmm(head_tail_emb, mention.unsqueeze(-1)).squeeze(-1)


class ConcatLinearDot(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.scaling = args.entity_dim ** -0.5
        self.linear = nn.Linear(2 * args.entity_dim, args.entity_dim)
        self.use_sentence_negatives = args.use_sentence_negatives

    def forward(self, mention, head, tail):
        head_tail_emb = torch.cat([head, tail], axis=-1)
        head_tail_emb = self.linear(head_tail_emb)
        if self.use_sentence_negatives:
            return self.scaling * torch.matmul(head_tail_emb.squeeze(1), mention.t())
        else:
            return self.scaling * torch.bmm(head_tail_emb, mention.unsqueeze(-1)).squeeze(-1)
