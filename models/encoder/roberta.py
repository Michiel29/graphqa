import torch
from fairseq.models.roberta import RobertaModel, RobertaEncoder
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from models.encoder.encoder_heads import encoder_head_dict

class RobertaWrapper(RobertaModel):

    def __init__(self, args, task):
        super().__init__(args, task)

        pretrain_encoder_path = getattr(args, 'pretrain_encoder_path', None)
        if pretrain_encoder_path is not None:
            self.load_from_pretrained(pretrain_encoder_path, args)
        self.padding_idx = task.dictionary.pad()
        self.head_idx = task.dictionary.head()
        self.tail_idx = task.dictionary.tail()

        self.custom_output_layer = encoder_head_dict[args.encoder_output_layer_type](args)

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
            padding_mask = (src_tokens != self.padding_idx) & (src_tokens != self.head_idx) & (src_tokens != self.tail_idx)
            x = self.custom_output_layer(x, padding_mask)
        return x, extra

    def load_from_pretrained(self, filename, args):

        state_dict = load_checkpoint_to_cpu(filename)['model']

        model_vocab_size = self.decoder.sentence_encoder.embed_tokens.weight.shape[0]
        ckpt_vocab_size = state_dict['decoder.sentence_encoder.embed_tokens.weight'].shape[0]
        diff = model_vocab_size - ckpt_vocab_size

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
            else:
                new_state_dict[n] = c

        missing_keys, unexpected_keys = super().load_state_dict(new_state_dict, strict=False, args=args)
        print('missing_keys: {}'.format(missing_keys))


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

def large_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    base_architecture(args)
