import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel

import tasks
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture
from modules.mlp import MLP_factory


@register_model('encoder_pmtb')
class EncoderPMTBModel(BaseFairseqModel):

    def __init__(self, args, encoder):
        super().__init__()

        self.args = args

        self.encoder_embed_dim = args.encoder_embed_dim
        self.encoder = encoder
        if self.args.scoring_function == 'linear':
            self.linear = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        elif self.args.scoring_function == 'mlp':
            self.mlp = MLP_factory([[2*args.encoder_embed_dim, 1]] + args.mlp_layer_sizes, layer_norm=args.mlp_layer_norm)

        self._max_positions = args.max_positions

    def max_positions(self):
        return self._max_positions

    def forward(self, batch):
        n_pairs = int(batch['A2B'].numel()/batch['size'])

        textA_enc, _ = self.encoder(batch['textA']) # [batch_size, enc_dim]
        textA_enc = torch.repeat_interleave(textA_enc, n_pairs, dim=0)

        textB_enc = []
        for cluster_texts in batch['textB'].values():
            cur_textB_enc, _ = self.encoder(cluster_texts)
            textB_enc.append(cur_textB_enc)
        textB_enc = torch.cat(textB_enc, dim=0)
        textB_enc = torch.index_select(textB_enc, 0, batch['A2B'])
        
        if self.args.scoring_function == 'linear':
            textB_enc = self.linear(textB_enc)
        elif self.args.scoring_function == 'mlp':
            textAB_enc = torch.cat((textA_enc, textB_enc), dim=-1)
            scores = self.mlp(textAB_enc).reshape(-1, n_pairs)
            return scores

        scores = (textA_enc * textB_enc).sum(dim=-1)
        scores = scores.reshape(-1, n_pairs)
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


@register_model_architecture('encoder_pmtb', 'encoder_pmtb__roberta_base')
def pmtb_base_architecture(args):
    base_architecture(args)


@register_model_architecture('encoder_pmtb', 'encoder_pmtb__roberta_large')
def pmtb_large_architecture(args):
    large_architecture(args)


@register_model_architecture('encoder_pmtb', 'encoder_pmtb__roberta_small')
def pmtb_small_architecture(args):
    small_architecture(args)
