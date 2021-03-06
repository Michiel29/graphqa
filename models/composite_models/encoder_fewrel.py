import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel, roberta
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture


@register_model('encoder_fewrel')
class EncoderFewRelModel(BaseFairseqModel):

    def __init__(self, args, encoder):
        super().__init__()

        self.encoder = encoder
        self.enc_dim = args.encoder_embed_dim
        self.n_way = args.n_way
        self.n_shot = args.n_shot

    def forward(self, batch):

        goal_text = batch['text'] # [batch_size, n_tokens]
        exemplars = batch['exemplars'] # [batch_size, n_way * n_shot, n_tokens]
        batch_size = batch['batch_size']

        goal_enc, _ = self.encoder(goal_text, annotation=batch.get('annotation'))
        goal_enc = goal_enc.unsqueeze(-1) # [batch_size, enc_dim, 1]
        exemplar_encs, _ = self.encoder(exemplars, annotation=batch.get('exemplars_annotation')) # [batch_size, n_way * n_shot, enc_dim]
        reshaped_exemplar_encs = torch.reshape(exemplar_encs, (batch_size, self.n_way, self.n_shot, -1))

        class_encs = torch.mean(reshaped_exemplar_encs, dim=2) # [batch_size, n_way, enc_dim]

        scores = torch.matmul(class_encs, goal_enc).squeeze(-1) # [batch_size, n_way]
        return scores

    @classmethod
    def build_model(cls, args, task, encoder=None):
        if encoder is None:
            encoder = RobertaWrapper.build_model(args, task)
        return cls(args, encoder)

@register_model_architecture('encoder_fewrel', 'encoder_fewrel__roberta_small')
def fewrel_small_architecture(args):
    small_architecture(args)

@register_model_architecture('encoder_fewrel', 'encoder_fewrel__roberta_base')
def fewrel_base_architecture(args):
    base_architecture(args)

@register_model_architecture('encoder_fewrel', 'encoder_fewrel__roberta_large')
def roberta_large_architecture(args):
    large_architecture(args)
