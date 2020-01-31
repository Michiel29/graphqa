import torch
from fairseq.models.roberta import RobertaModel

class RobertaWrapper(RobertaModel):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.output_layer_dict = {'bag_of_words': self.bag_of_words}

    def forward(self, src_tokens, layer_type, features_only=False, return_all_hiddens=False, masked_tokens=None, **unused):
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
            x = self.output_layer(x, layer_type)
        return x, extra
    
    def output_layer(self, x, output_layer_type):
        return self.output_layer_dict[output_layer_type](x)
    
    def bag_of_words(self, x):
        return torch.mean(x, dim=-2)


