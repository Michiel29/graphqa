import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel

import tasks
from models.triplet import triplet_dict
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture
from utils.diagnostic_utils import Diagnostic

@register_model('encoder_triplet')
class EncoderTripletModel(BaseFairseqModel):

    def __init__(self, args, encoder, triplet_model, n_entities, task):
        super().__init__()

        self.args = args

        self.entity_dim = args.entity_dim
        self.encoder_embed_dim = args.encoder_embed_dim

        self.encoder = encoder
        self.entity_embedder = nn.Embedding(n_entities, args.entity_dim)
        self.triplet_model = triplet_model

        self.task = task
        self._max_positions = args.max_positions

    def max_positions(self):
        return self._max_positions

    def forward(self, batch):

        text_enc, _ = self.encoder(batch['text']) # [batch_size, enc_dim]

        head_emb = self.entity_embedder(batch['head']) # [batch_size, (1 + k_negative), ent_dim]
        tail_emb = self.entity_embedder(batch['tail']) # [batch_size, (1 + k_negative), ent_dim]

        multiply_view = [-1] * len(head_emb.shape)
        multiply_view[-2] = head_emb.shape[-2]
        text_enc = text_enc.unsqueeze(-2).expand(multiply_view) # [batch_size, (1 + k_negative), ent_dim]

        scores = self.triplet_model(text_enc, head_emb, tail_emb) # [batch_size, (1 + k_negative)]

        #diag = Diagnostic(self.task.dictionary, self.task.entity_dictionary, self.task)
        #diag.inspect_batch(batch, scores=scores)

        return scores

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        parser.add_argument('--triplet_type', type=str, default=None,
                            help='type of triplet model to use for inference')

        RobertaWrapper.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        encoder = RobertaWrapper.build_model(args, task)
        triplet_model = triplet_dict[args.triplet_type](args)
        n_entities = len(task.entity_dictionary)

        return cls(args, encoder, triplet_model, n_entities, task)

@register_model_architecture('encoder_triplet', 'encoder_triplet__roberta_base')
def triplet_base_architecture(args):
    base_architecture(args)

@register_model_architecture('encoder_triplet', 'encoder_triplet__roberta_large')
def triplet_large_architecture(args):
    large_architecture(args)

@register_model_architecture('encoder_triplet', 'encoder_triplet__roberta_small')
def triplet_small_architecture(args):
    small_architecture(args)
