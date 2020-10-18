import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel

import tasks
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture


@register_model('encoder_etp_relation')
class EncoderETPRelation(BaseFairseqModel):

    def __init__(self, args, encoder, n_entities):
        super().__init__()

        self.args = args

        self.entity_dim = args.entity_dim
        self.encoder_embed_dim = args.encoder_embed_dim

        self.encoder = encoder
        self.entity_embedder = nn.Embedding(n_entities, args.entity_dim)

    def forward(self, batch):

        mention_enc, _ = self.encoder(batch['text'], annotation=batch.get('annotation')) # [batch_size, enc_dim]

        candidate_embeddings = self.entity_embedder(batch['candidates']) # [batch_size, k_candidates, ent_dim]

        scores = (mention_enc.unsqueeze(1) * candidate_embeddings).sum(axis=-1)
        return scores

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        parser.add_argument('--triplet_type', type=str, default=None,
                            help='type of triplet model to use for inference')

        RobertaWrapper.add_args(parser)

    @classmethod
    def build_model(cls, args, task, encoder=None):
        if encoder is None:
            encoder = RobertaWrapper.build_model(args, task)
        n_entities = len(task.entity_dictionary)
        return cls(args, encoder, n_entities)


@register_model_architecture('encoder_etp_relation', 'encoder_etp_relation__roberta_base')
def triplet_base_architecture(args):
    base_architecture(args)


@register_model_architecture('encoder_etp_relation', 'encoder_etp_relation__roberta_large')
def triplet_large_architecture(args):
    large_architecture(args)


@register_model_architecture('encoder_etp_relation', 'encoder_etp_relation__roberta_small')
def triplet_small_architecture(args):
    small_architecture(args)
