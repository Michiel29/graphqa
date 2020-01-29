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
        self.enc_dim = args.enc_dim

    def forward(self, batch):

        goal_mention = batch['goal_mention'] # [batch_size, n_tokens]
        candidate_mentions = batch['candidate_mentions'] # [batch_size, n_way, n_shot, n_tokens]
        batch_size = goal_mention.shape[0] 
       
        goal_enc = self.encoder(goal_mention).unsqueeze(-1) # [batch_size, enc_dim, 1]
        candidate_encs = self.encoder(candidate_mentions) # [batch_size, n_way, n_shot, enc_dim]
        rel_prototypes = torch.mean(candidate_encs, dim=-2) # [batch_size, n_way, enc_dim]

        scores = torch.matmul(rel_prototypes, goal_enc).squeeze(-1) # [batch_size, n_way]

        return scores

    @classmethod
    def build_model(cls, args, task):
        
        encoder = encoder_dict[args.encoder_type](args)

        return cls(args, encoder)

