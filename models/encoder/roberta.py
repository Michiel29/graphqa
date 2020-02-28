import torch
from torch import nn

from fairseq.models.roberta import RobertaModel, base_architecture
from fairseq.models.roberta import roberta_large_architecture as large_architecture
from fairseq.models import register_model, register_model_architecture
from fairseq.checkpoint_utils import load_checkpoint_to_cpu

from models.encoder.encoder_heads import encoder_head_dict
from utils.checkpoint_utils import handle_state_dict_keys


@register_model('roberta_wrapper')
class RobertaWrapper(RobertaModel):

    def __init__(self, args, encoder):
        super().__init__(args, encoder)

        pretrain_encoder_path = getattr(args, 'pretrain_encoder_path', None)
        if pretrain_encoder_path is not None:
            self.load_from_pretrained(pretrain_encoder_path, args)

        encoder_head_dropout_rate = getattr(args, 'encoder_head_dropout', 0)
        if encoder_head_dropout_rate > 0:
            self.encoder_head_dropout = nn.Dropout(p=encoder_head_dropout_rate)
        else:
            self.encoder_head_dropout = None

        if args.encoder_output_layer_type == 'lm_head':
            self.custom_output_layer = self.output_layer
        else:
            self.custom_output_layer = encoder_head_dict[args.encoder_output_layer_type](args, encoder.dictionary)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
        Returns:
            tuple:
                - x:
                    - bag_of_words: `(batch, encoder_embed_dim)`
                - extra:
                    a dictionary of additional data, where 'inner_states'
                    is a list of hidden states. Note that the hidden
                    states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)

        if not features_only:
            if self.encoder_head_dropout is not None:
                x = self.encoder_head_dropout(x)
            x = self.custom_output_layer(x, src_tokens=src_tokens, masked_tokens=masked_tokens)

        return x, extra

    def load_from_pretrained(self, filename, args):

        state_dict = load_checkpoint_to_cpu(filename)['model']

        model_vocab_size = self.decoder.sentence_encoder.embed_tokens.weight.shape[0]
        ckpt_vocab_size = state_dict['decoder.sentence_encoder.embed_tokens.weight'].shape[0]
        diff = model_vocab_size - ckpt_vocab_size

        model_pos_size = self.decoder.sentence_encoder.embed_positions.weight.shape[0]
        ckpt_pos_size = state_dict['decoder.sentence_encoder.embed_positions.weight'].shape[0]
        diff_pos_size = model_pos_size - ckpt_pos_size

        new_state_dict = {}
        for n, c in state_dict.items():
            if n in ['decoder.sentence_encoder.embed_tokens.weight', 'decoder.lm_head.weight'] and diff > 0:
                new_weight = torch.Tensor(c.shape[0]+diff, c.shape[1])
                new_weight.data.normal_(mean=0.0, std=0.02)
                new_weight[:-diff] = c
                new_state_dict[n] = new_weight
            elif n == 'decoder.lm_head.bias' and diff > 0:
                new_weight = torch.zeros(c.shape[0]+diff)
                new_weight[:-diff] = c
                new_state_dict[n] = new_weight
            elif n == 'decoder.sentence_encoder.embed_positions.weight' and diff_pos_size < 0:
                new_weight = c[:c.shape[0] + diff_pos_size]
                new_state_dict[n] = new_weight
            else:
                new_state_dict[n] = c

        missing_keys, unexpected_keys = super().load_state_dict(new_state_dict, strict=False, args=args)
        handle_state_dict_keys(missing_keys, unexpected_keys)


@register_model_architecture('roberta_wrapper', 'roberta_small')
def small_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    base_architecture(args)

@register_model_architecture('roberta_wrapper', 'roberta_wrapper_base')
def base_wrapper_architecture(args):
    base_architecture(args)