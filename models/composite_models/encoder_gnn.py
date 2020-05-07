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
        self.mlp = MLP_factory(args.layer_sizes)
        self.gnn_layers = nn.ModuleList([gnn_layer_dict[args.gnn_layer_type](args.gnn_layer_args) for i in range(args.gnn_layer_args['n_gnn_layers']-1)])
        self.neg_type = args.neg_type

    def encode_text(self, text_chunks):
        text_enc = []
        for chunk in text_chunks:
            text_enc.append(self.encoder(chunk))
        text_enc = torch.cat(text_enc, dim=0)
        return text_enc

    def forward(self, batch):

        batch = {}

        # batch = {
        #   'text': list of n_chunks token tensors, sorted by ascending length -- shape (chunk_size, chunk_text_len)
        #   'graph': list of tensors which are indices into text_enc -- len(batch['graph']) = n_targets, shape of each tensor is (m_i, 2)
        #   'target_text_idx': indices into text_enc -- shape (n_targets)
        #   'target': torch.arange(n_targets)
        # }

        text_enc = self.encode_text(batch['text'])
        device = text_enc.device

        target_text_idx = batch['target_text_idx']
        n_targets = len(target_text_idx)
        n_matches = n_targets ** 2

        graph_sizes = torch.LongTensor([len(g) for g in batch['graph']], device=device) # (n_targets)

        graph_idx = torch.cat(batch['graph'], dim=0) # (sum(m_i), 2)
        graph_idx = graph_idx.unsqueeze(0).expand(n_targets, -1, -1).reshape(-1) # (n_targets * sum(m_i) * 2)

        target_text_idx_expand = target_text_idx.unsqueeze(-1).expand(-1, n_targets).reshape(-1) # (n_targets ** 2)
        graph_sizes_expand = graph_sizes.unsqueeze(0).expand(n_targets, -1).reshape(-1) # (n_targets ** 2)
        
        target_idx_range = torch.arange(n_targets, device=device).unsqueeze(-1).expand(-1, n_targets).reshape(-1) # (n_targets ** 2)
        put_indices = tuple(torch.repeat_interleave(target_idx_range, graph_sizes_expand, dim=0).unsqueeze(0)) # (n_targets * sum(m_i))

        graph_rep = text_enc[graph_idx].reshape(len(graph_idx)/2, 2, -1) # (n_targets * sum(m_i), 2, d)
        target_rep = text_enc[target_text_idx_expand] # (n_targets ** 2, d)  

        for layer in self.gnn_layers:
            target_rep_repeat = torch.repeat_interleave(target_rep, graph_sizes_expand, dim=0) # (n_targets * sum(m_i), d)
            layer_output = layer(target_rep_repeat, graph_rep) # (n_targets * sum(m_i), d)
            target_rep = target_text_idx_expand.index_put(put_indices, layer_output, accumulate=True) # (n_targets ** 2, d)

        scores = self.mlp(target_rep) # (n_targets ** 2)
        scores = scores.reshape(n_targets, n_targets) # (n_targets, n_targets) -- rows=texts, cols=graphs
        
        if self.neg_type == 'graph':
            pass
        elif self.neg_type == 'text':
            scores = scores.t()
        elif self.neg_type == 'graph_text':
            scores_graph = scores
            mask = 1 - torch.eye(n_targets)
            scores_text = torch.masked_select(scores_graph.t(), mask).reshape(n_targets, n_targets-1) # (n_targets, n_target-1)
            scores = torch.cat((scores_graph, scores_text), dim=1) # (n_targets, 2*n_targets-1)
        else:
            raise Exception('neg_type {} does not exist'.format(self.neg_type))

        return scores

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        parser.add_argument('--gnn_type', type=str, default='nlm',
                            help='type of gnn model to use for inference')

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