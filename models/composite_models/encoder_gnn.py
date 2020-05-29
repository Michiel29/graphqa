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

        if args.placeholder_input:
            self.placeholder_input = nn.Parameter(torch.ones(args.entity_dim))
        else:
            self.mlp = MLP_factory([[args.gnn_layer_args['enc_dim'], 1]] + args.layer_sizes, layer_norm=args.gnn_mlp_layer_norm)


    def encode_text(self, text_chunks):
        text_enc = []
        for chunk in text_chunks:
            text_enc.append(self.encoder(chunk)[0])
        text_enc = torch.cat(text_enc, dim=0)
        return text_enc


    def forward(self, batch):
        # batch = {
        #   'text': list of n_chunks token tensors, sorted by ascending length -- shape (chunk_size, chunk_text_len)
        #   'graph': Tensor of indices into text_enc. shape of tensor is (sum m_i, 2)
        #   'graph_sizes': tensor of n_targets * n_candidates, length of each graph. Within target should be same len
        #   'candidate_text_idx': for each neighborhood, contains all candidates sentence idx including target sentences- tensor of shape n_targets by n_candidates
        #   'target': torch.zeros(n_targets)
        # }

        text_enc = self.encode_text(batch['text'])
        device = text_enc.device

        candidate_text_idx = batch['candidate_text_idx']
        n_candidates = candidate_text_idx.shape[-1] # (n_targets, n_candidates)
        n_targets = len(candidate_text_idx)

        graph_sizes = batch['graph_sizes']

        graph_idx = batch['graph']

        candidate_text_idx = candidate_text_idx.reshape(-1) # (n_targets * n_candidates)

        candidate_idx_range = torch.arange(n_targets * n_candidates, device=device) # (n_targets * n_candidates)
        put_indices = tuple(torch.repeat_interleave(candidate_idx_range, graph_sizes, dim=0).unsqueeze(0)) # (sum(m_i))

        graph_rep = text_enc[graph_idx] # (sum(m_i), 2, d)
        candidate_rep = text_enc[candidate_text_idx] # (n_targets * n_candidates, d)

        if self.args.placeholder_input:
            layer_candidate_rep = self.placeholder_input.unsqueeze(0).expand(n_targets * n_candidates, -1)
        else:
            layer_candidate_rep = candidate_rep


        for layer in self.gnn_layers:
            layer_candidate_rep, graph_rep = layer(layer_candidate_rep, graph_rep,
            graph_sizes, put_indices) # (sum(m_i), d)

        if self.args.placeholder_input:
            scores = (layer_candidate_rep * candidate_rep).sum(dim=-1)
        else:
            scores = self.mlp(layer_candidate_rep) # (n_targets * n_candidates)
        scores = scores.reshape(n_targets, n_candidates) # (n_targets, n_candidates)

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