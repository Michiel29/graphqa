import torch

from fairseq.models import register_model, register_model_architecture

from models.composite_models.encoder_gnn import EncoderGNNModel
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture

@register_model('encoder_eval')
class EncoderGNNEval(EncoderGNNModel):

    def forward(self, batch):
        text_enc = self.encode_text(batch['text'])
        device = text_enc.device

        target_text_idx = batch['target_text_idx']
        n_targets = len(target_text_idx)
        graph_idx = batch['graph']
        graph_sizes = batch['graph_sizes']
        batch_size = batch['size']

        target_text_idx = target_text_idx.reshape(-1) # (n_targets)

        target_idx_range = torch.arange(n_targets, device=device) # (n_targets)
        put_indices = tuple(torch.repeat_interleave(target_idx_range, graph_sizes, dim=0).unsqueeze(0)) # (sum(m_i))

        graph_rep = text_enc[graph_idx] # (sum(m_i), 2, d)
        target_rep = text_enc[target_text_idx] # (n_targets, d)

        if self.args.placeholder_input:
            layer_target_rep = self.placeholder_input.unsqueeze(0).expand(n_targets, -1)
        else:
            layer_target_rep = target_rep

        for layer in self.gnn_layers:
            layer_target_rep, graph_rep = layer(layer_target_rep, graph_rep,
            graph_sizes, put_indices) # (sum(m_i), d)

        if self.args.placeholder_input:
            scores = (layer_target_rep * target_rep).sum(dim=-1)
        else:
            scores = self.mlp(layer_target_rep) # (n_targets * n_candidates)
        scores = scores.reshape(batch_size, -1) # (batch_size, n_texts)

        return scores


@register_model_architecture('encoder_eval', 'encoder_eval__roberta_small')
def encoder_eval_small_architecture(args):
    small_architecture(args)