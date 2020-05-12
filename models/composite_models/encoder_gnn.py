import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel, roberta

import tasks

from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture
from models.gnn import gnn_layer_dict
from modules.mlp import MLP_factory


@register_model('encoder_gnn')
class EncoderGNNModel(BaseFairseqModel):

    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.gnn_layers = nn.ModuleList([
            gnn_layer_dict[args.gnn_layer_type](args.gnn_layer_args)
            for i in range(args.gnn_layer_args['n_gnn_layers'])
        ])
        final_gnn_layer_dim = args.gnn_layer_args['layer_sizes'][-1][0]
        self.mlp = MLP_factory([[final_gnn_layer_dim, 1]] + args.layer_sizes, layer_norm=args.gnn_mlp_layer_norm)
        self.neg_type = args.neg_type

    def encode_text(self, text_chunks):
        text_enc = []
        for chunk in text_chunks:
            text_enc.append(self.encoder(chunk)[0])
        text_enc = torch.cat(text_enc, dim=0)
        return text_enc

    def forward(self, batch):
        # batch = {
        #   'text': list of n_chunks token tensors, sorted by ascending length -- shape (chunk_size, chunk_text_len)
        #   'graph': list of tensors which are indices into text_enc -- len(batch['graph']) = n_targets, shape of each tensor is (m_i, 2)
        #   'target_text_idx': indices into text_enc -- shape (n_targets)
        #   'candidate_idx': for each neighborhood, contains all candidates sentence idx including target sentences- tensor of shape n_targets by n_candidates
        #   'target': torch.arange(n_targets)
        # }

        text_enc = self.encode_text(batch['text'])
        device = text_enc.device

        target_text_idx = batch['target_text_idx']
        n_targets = len(target_text_idx)
        n_matches = n_targets ** 2
        #   candidate_idx = batch['candidates']

        candidate_idx = target_text_idx.unsqueeze(0).expand(n_targets, -1)
        n_candidates = candidate_idx.shape[-1]

        graph_sizes = torch.tensor([len(g) for g in batch['graph']], dtype=torch.int64, device=device) # (n_targets)

        graph_idx = torch.cat(batch['graph'], dim=0) # (sum(m_i), 2)
        graph_idx = graph_idx.unsqueeze(0).expand(n_candidates, -1, -1).reshape(-1) # (n_candidates * sum(m_i) * 2)

        candidate_idx_transpose = candidate_idx.t().reshape(-1) # (n_targets * n_candidates)

        graph_sizes_expand = graph_sizes.unsqueeze(0).expand(n_candidates, -1).reshape(-1) # (n_targets * n_candidates)

        candidate_idx_range = torch.arange(n_targets * n_candidates, device=device) # (n_targets * n_candidates)
        put_indices = tuple(torch.repeat_interleave(candidate_idx_range, graph_sizes_expand, dim=0).unsqueeze(0)) # (n_targets * sum(m_i))

        assert len(graph_idx) % 2 == 0
        graph_rep = text_enc[graph_idx].reshape(len(graph_idx) // 2, 2, -1) # (n_targets * sum(m_i), 2, d)
        candidate_rep = text_enc[candidate_idx_transpose].unsqueeze(1) # (n_targets * n_candidates, 1, d)

        for layer in self.gnn_layers:
            candidate_rep_repeat = torch.repeat_interleave(candidate_rep, graph_sizes_expand, dim=0) # (n_candidates * sum(m_i), 1, d)
            layer_output = layer(candidate_rep_repeat, graph_rep).unsqueeze(-2) # (n_candidates * sum(m_i), d)
            candidate_rep = candidate_rep.index_put(put_indices, layer_output, accumulate=True) # (n_targets * n_candidates, d)

        scores = self.mlp(candidate_rep) # (n_targets * n_candidates)
        scores = scores.reshape(n_candidates, n_targets) # (n_candidates, n_targets)
        scores = scores.t() # (n_targets, n_candidates)

        # if self.neg_type == 'text':
        #     pass
        # elif self.neg_type == 'text':
        #     scores = scores.t()
        # elif self.neg_type == 'graph_text':
        #     scores_graph = scores
        #     mask = (1 - torch.eye(n_targets, device=device)).bool()
        #     scores_text = torch.masked_select(scores_graph.t(), mask).reshape(n_targets, n_targets - 1) # (n_targets, n_target - 1)
        #     scores = torch.cat((scores_graph, scores_text), dim=1) # (n_targets, 2*n_targets-1)
        # else:
        #     raise Exception('neg_type {} does not exist'.format(self.neg_type))

        return scores

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        parser.add_argument('--gnn-layer-type', type=str, default=None)

    @classmethod
    def build_model(cls, args, task, encoder=None):
        if encoder is None:
            encoder = RobertaWrapper.build_model(args, task)
        return cls(args, encoder)


@register_model_architecture('encoder_gnn', 'encoder_gnn__roberta_base')
def encoder_gnn_base_architecture(args):
    base_architecture(args)


@register_model_architecture('encoder_gnn', 'encoder_gnn__roberta_large')
def encoder_gnn_large_architecture(args):
    large_architecture(args)


@register_model_architecture('encoder_gnn', 'encoder_gnn__roberta_small')
def encoder_gnn_small_architecture(args):
    small_architecture(args)