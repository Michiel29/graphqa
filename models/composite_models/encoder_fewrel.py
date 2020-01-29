import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel, roberta


import tasks

from models.encoder import encoder_dict

@register_model('encoder_fewrel')
class EncoderFewRelModel(BaseFairseqModel):

    def __init__(self, args, encoder):
        super().__init__()

        self.args = args

        self.encoder = encoder

        self.n_way = args.n_way
        self.enc_dim = args.enc_dim

    def forward(self, batch):

        goal_mention = batch['goal_mention']
        candidate_mentions = batch['candidate_mentions']
        batch_size = goal_mention.shape[0] 
       
        rel_prototypes = torch.zeros(batch_size, self.n_way, self.enc_dim)
        for idx, rel in enumerate(candidate_mentions.keys()):
            candidate_mention_encs = self.encoder(candidate_mentions[rel]) # [batch_size, n_way, n_shot, enc_dim]
            rel_prototypes[:, idx, :]  = torch.mean(candidate_mention_encs, dim=-2) # [batch_size, n_way, enc_dim]

        goal_enc = self.encoder(goal_mention).unsqueeze(-1) # [batch_size, enc_dim, 1]

        scores = torch.matmul(rel_prototypes, goal_enc).squeeze(-1) # [batch_size, n_way]

        return scores

    @classmethod
    def build_model(cls, args, task):
        
        encoder = encoder_dict[args.encoder_type](args)

        return cls(args, encoder)

