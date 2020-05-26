import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel, roberta
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture


@register_model('encoder_tacred')
class EncoderTACREDModel(BaseFairseqModel):

    def __init__(self, args, encoder):
        super().__init__()

        self.encoder = encoder
        if args.encoder_output_layer_type in ['entity_start', 'entity_start_layer_norm', 'entity_pooling_first_token']:
            self.classifier = nn.Linear(2*args.encoder_embed_dim, args.num_classes)
        elif args.encoder_output_layer_type in ['entity_start_linear', 'entity_start_mlp']:
            self.classifier = nn.Linear(args.entity_dim, args.num_classes)
        else:
            self.classifier = nn.Linear(args.encoder_embed_dim, args.num_classes)

    def forward(self, batch):
        text = batch['text'] # [batch_size, n_tokens]
        text_enc, _ = self.encoder(text, annotation=batch.get('annotation'))
        scores = self.classifier(text_enc)
        return scores

    @classmethod
    def build_model(cls, args, task, encoder=None):
        if encoder is None:
            encoder = RobertaWrapper.build_model(args, task)
        return cls(args, encoder)


@register_model_architecture('encoder_tacred', 'encoder_tacred__roberta_small')
def roberta_small_architecture(args):
    small_architecture(args)


@register_model_architecture('encoder_tacred', 'encoder_tacred__roberta_base')
def roberta_base_architecture(args):
    base_architecture(args)


@register_model_architecture('encoder_tacred', 'encoder_tacred__roberta_large')
def roberta_large_architecture(args):
    large_architecture(args)
