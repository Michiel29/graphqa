import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel

import tasks
from models.triplet import triplet_dict
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture


@register_model('encoder_triplet')
class EncoderTripletModel(BaseFairseqModel):

    def __init__(self, args, encoder, triplet_model, n_entities):
        super().__init__()

        self.args = args

        self.entity_dim = args.entity_dim
        self.encoder_embed_dim = args.encoder_embed_dim

        self.encoder = encoder
        self.entity_embedder = nn.Embedding(n_entities, args.entity_dim)
        self.mention_linear = nn.Linear(args.encoder_embed_dim, args.entity_dim)
        self.triplet_model = triplet_model

    def forward(self, batch):
        mention_enc, _ = self.encoder(batch['mention'])
        mention_enc = self.mention_linear(mention_enc)

        head_emb = self.entity_embedder(batch['head'])
        tail_emb = self.entity_embedder(batch['tail'])

        multiply_view = [-1] * len(head_emb.shape)
        multiply_view[-2] = head_emb.shape[-2]
        mention_enc = mention_enc.unsqueeze(-2).expand(multiply_view)

        score = self.triplet_model(mention_enc, head_emb, tail_emb)
        
        return score

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        parser.add_argument('--triplet_type', type=str, default='distmult',
                            help='type of triplet model to use for inference')

        RobertaWrapper.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        
        encoder = RobertaWrapper.build_model(args, task)
        triplet_model = triplet_dict[args.triplet_type](args)
        n_entities = len(task.entity_dictionary)

        return cls(args, encoder, triplet_model, n_entities)

@register_model_architecture('encoder_triplet', 'encoder_triplet__roberta_base')
def triplet_base_architecture(args):
    base_architecture(args)


@register_model_architecture('encoder_triplet', 'encoder_triplet__roberta_large')
def roberta_large_architecture(args):
    large_architecture




