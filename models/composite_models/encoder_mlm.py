import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel

import tasks
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture


@register_model('encoder_mlm')
class EncoderMLMModel(BaseFairseqModel):

    def __init__(self, args, encoder):
        super().__init__()

        self.encoder = encoder
        self._max_positions = args.max_positions

    def max_positions(self):
        return self._max_positions

    def forward(self, batch):
        text_enc, _ = self.encoder(batch) # [batch_size, enc_dim]
        return text_enc

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        RobertaWrapper.add_args(parser)

    @classmethod
    def build_model(cls, args, task, encoder=None):
        if encoder is None:
            encoder = RobertaWrapper.build_model(args, task)
        return cls(args, encoder)


@register_model_architecture('encoder_mlm', 'encoder_mlm__roberta_base')
def mtb_base_architecture(args):
    base_architecture(args)


@register_model_architecture('encoder_mlm', 'encoder_mlm__roberta_large')
def mtb_large_architecture(args):
    large_architecture(args)


@register_model_architecture('encoder_mlm', 'encoder_mlm__roberta_small')
def mtb_small_architecture(args):
    small_architecture(args)
