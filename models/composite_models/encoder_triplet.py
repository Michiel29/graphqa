import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel

import tasks
from models.triplet import triplet_dict
from models.encoder.roberta import RobertaWrapper


@register_model('encoder_triplet')
class EncoderTripletModel(BaseFairseqModel):

    def __init__(self, args, encoder, triplet_model, n_entities):
        super().__init__()

        self.args = args

        self.entity_dim = args.entity_dim
        self.encoder_embed_dim = args.encoder_embed_dim
        self.encoder_output_layer_type = args.encoder_output_layer_type

        self.encoder = encoder
        self.entity_embedder = nn.Embedding(n_entities, args.entity_dim)
        self.mention_linear = nn.Linear(args.encoder_embed_dim, args.entity_dim)
        self.triplet_model = triplet_model

    def forward(self, batch):
        mention_enc, _ = self.encoder(batch['mention'], self.encoder_output_layer_type)
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

    def load_state_dict(self, state_dict, strict, args):

        if 'encoder.decoder.sentence_encoder.embed_tokens.weight' in state_dict.keys():
            missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=True, args=args)

        else: 
            model_vocab_size = self.encoder.decoder.sentence_encoder.embed_tokens.weight.shape[0]
            ckpt_vocab_size = state_dict['decoder.sentence_encoder.embed_tokens.weight'].shape[0]
            diff = model_vocab_size - ckpt_vocab_size

            if diff > 0:
                new_state_dict = {}
                for n, c in state_dict.items():
                    name = 'encoder.' + n
                    if n in ['decoder.sentence_encoder.embed_tokens.weight', 'decoder.lm_head.weight']:
                        new_weight = torch.Tensor(c.shape[0]+diff, c.shape[1])
                        new_weight.data.normal_(mean=0.0, std=0.02)
                        new_weight[:-diff] = c
                        new_state_dict[name] = new_weight
                    elif n == 'decoder.lm_head.bias':
                        new_weight = torch.zeros(c.shape[0]+diff)
                        new_weight[:-diff] = c
                        new_state_dict[name] = new_weight
                    else:
                        new_state_dict[name] = c

                missing_keys, unexpected_keys = super().load_state_dict(new_state_dict, strict=False, args=args)
                print('missing_keys: {}'.format(missing_keys))
                print('unexpected_keys: {}'.format(unexpected_keys))

        return missing_keys, unexpected_keys



@register_model_architecture('encoder_triplet', 'encoder_triplet__roberta_base')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12) 
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12) 

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    args.encoder_layers_to_keep = getattr(args, 'encoder_layers_to_keep', None)
    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0.0)

@register_model_architecture('encoder_triplet', 'encoder_triplet__roberta_large')
def roberta_large_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24) 
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16) 
    base_architecture(args)



