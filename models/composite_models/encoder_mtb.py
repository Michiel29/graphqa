import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel

import tasks
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture

# from utils.diagnostic_utils import inspect_batch

@register_model('encoder_mtb')
class EncoderMTBModel(BaseFairseqModel):

    def __init__(self, args, encoder, task):
        super().__init__()

        self.args = args

        self.encoder_embed_dim = args.encoder_embed_dim
        self.encoder = encoder
        if args.encoder_output_layer_type == 'bag_of_words':
            self.mention_linear = nn.Linear(args.encoder_embed_dim, args.entity_dim)
        elif args.encoder_output_layer_type == 'head_tail_concat':
            self.mention_linear = nn.Linear(2 * args.encoder_embed_dim, args.entity_dim)

        self.task = task
        self._max_positions = args.max_positions

    def max_positions(self):
        return self._max_positions

    def forward(self, batch):

        mention1_enc, _ = self.encoder(batch['mention1']) # [batch_size, enc_dim]
        mention1_enc = self.mention_linear(mention1_enc) # [batch_size, ent_dim]

        mention2_enc, _ = self.encoder(batch['mention2']) # [batch_size, enc_dim]
        mention2_enc = self.mention_linear(mention2_enc) # [batch_size, ent_dim]
       
        scores = (mention1_enc * mention2_enc).sum(dim=-1) 

        #inspect_batch(batch, self.task, scores)

        return scores

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        RobertaWrapper.add_args(parser)

    @classmethod
    def build_model(cls, args, task):

        encoder = RobertaWrapper.build_model(args, task)
        n_entities = len(task.entity_dictionary)

        return cls(args, encoder, task)

@register_model_architecture('encoder_mtb', 'encoder_mtb__roberta_base')
def mtb_base_architecture(args):
    base_architecture(args)

@register_model_architecture('encoder_mtb', 'encoder_mtb__roberta_large')
def mtb_large_architecture(args):
    large_architecture(args)

@register_model_architecture('encoder_mtb', 'encoder_mtb__roberta_small')
def mtb_small_architecture(args):
    small_architecture(args)




