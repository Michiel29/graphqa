import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel

import tasks
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture
from utils.diagnostic_utils import Diagnostic


@register_model('encoder_dual')
class EncoderDualModel(BaseFairseqModel):

    def __init__(self, args, encoder, n_entities):
        super().__init__()

        self.args = args

        self.entity_dim = args.entity_dim
        self.encoder_embed_dim = args.encoder_embed_dim

        self.encoder = encoder
        self.entity_embedder = nn.Embedding(n_entities, args.entity_dim)
        self.scaling = self.entity_dim ** -0.5

        self._max_positions = args.max_positions

    def max_positions(self):
        return self._max_positions

    def forward(self, batch):
        replace_heads = torch.unsqueeze(batch['replace_heads'][:, 0], -1)
        text_enc, _ = self.encoder(batch['text'], replace_heads=replace_heads) # [batch_size, enc_dim]

        head_emb = self.entity_embedder(batch['head']) # [batch_size, (1 + k_negative), ent_dim]
        tail_emb = self.entity_embedder(batch['tail']) # [batch_size, (1 + k_negative), ent_dim]

        head_or_tail = torch.unsqueeze(replace_heads, -1) * head_emb + (1 - torch.unsqueeze(replace_heads, -1)) * tail_emb

        return self.scaling * torch.bmm(head_or_tail, text_enc.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        RobertaWrapper.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        encoder = RobertaWrapper.build_model(args, task)
        n_entities = len(task.entity_dictionary)
        return cls(args, encoder, n_entities)

@register_model_architecture('encoder_dual', 'encoder_dual__roberta_base')
def triplet_base_architecture(args):
    base_architecture(args)

@register_model_architecture('encoder_dual', 'encoder_dual__roberta_large')
def triplet_large_architecture(args):
    large_architecture(args)

@register_model_architecture('encoder_dual', 'encoder_dual__roberta_small')
def triplet_small_architecture(args):
    small_architecture(args)
