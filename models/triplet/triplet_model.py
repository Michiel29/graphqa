import torch
import torch.nn as nn
import torch.nn.functional as F

from models.triplet import triplet_dict

class TripletModel(nn.Module):
    def __init__(self, triplet_type, n_entities, entity_dim, args):
        super().__init__()
        self.emb = nn.Embedding(n_entities, entity_dim)
        self.triplet_model = triplet_dict[triplet_type](args)

    def forward(self, mention, h, t):
        h_emb = self.emb(h)
        t_emb = self.emb(t)

        self.score = self.triplet_model(h_emb, mention, t_emb)
        

