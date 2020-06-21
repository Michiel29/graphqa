import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel

import tasks
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture
from modules.mlp import MLP_factory


@register_model('encoder_graph_distance')
class EncoderGraphDistance(BaseFairseqModel):

    def __init__(self, args, encoder):
        super().__init__()

        self.args = args

        self.encoder_embed_dim = args.encoder_embed_dim
        self.encoder = encoder
        self.mlp = MLP_factory(args.layer_sizes, layer_norm=True)



    def forward(self, batch):

        textA_enc, _ = self.encoder(batch['textA']) # [batch_size, enc_dim]

        textB_enc = []
        for cluster_texts in batch['textB'].values():
            cur_textB_enc, _ = self.encoder(cluster_texts)
            textB_enc.append(cur_textB_enc)
        textB_enc = torch.cat(textB_enc, dim=0)
        textB_enc = torch.index_select(textB_enc, 0, batch['A2B'])
        scores = self.mlp(torch.cat((textA_enc, textB_enc), dim=1))

        return scores

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        RobertaWrapper.add_args(parser)

    @classmethod
    def build_model(cls, args, task, encoder=None):
        if encoder is None:
            encoder = RobertaWrapper.build_model(args, task)
        return cls(args, encoder)


@register_model_architecture('encoder_graph_distance', 'encoder_graph_distance__roberta_base')
def graph_distance_base_architecture(args):
    base_architecture(args)


@register_model_architecture('encoder_graph_distance', 'encoder_graph_distance__roberta_large')
def graph_distance_large_architecture(args):
    large_architecture(args)


@register_model_architecture('encoder_graph_distance', 'encoder_graph_distance__roberta_small')
def graph_distance_small_architecture(args):
    small_architecture(args)
