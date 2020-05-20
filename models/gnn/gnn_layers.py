from torch import nn
import torch
from modules.mlp import MLP_factory


class MLPConcatLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.mlp = MLP_factory([[3 * args['enc_dim'], 1]] + args['layer_sizes'], layer_norm=args['layer_norm'])

    def forward(self, target, graph, **unused):
        # target: (n_targets * sum(m_i), 1, d)
        # graph: (n_targets * sum(m_i), 2, d)

        enc_dim = target.shape[-1]
        target = target.reshape(-1, enc_dim)
        graph = graph.reshape(-1, 2 * enc_dim)

        return self.mlp(torch.cat((target, graph), dim=-1))


class MLPLinearConcatLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.graph_linear = nn.Linear(2*args['enc_dim'], args['linear_dim'])
        self.mlp = MLP_factory([[2 * args['enc_dim'], 1]] + args['layer_sizes'], layer_norm=args['layer_norm'])

    def forward(self, target, graph, **unused):
        # target: (n_targets * sum(m_i), 1, d)
        # graph: (n_targets * sum(m_i), 2, d)

        enc_dim = target.shape[-1]
        target = target.reshape(-1, enc_dim)
        graph = graph.reshape(-1, 2*enc_dim)
        graph_emb = self.graph_linear(graph)

        return self.mlp(torch.cat((target, graph_emb), dim=-1))


class MLPConcatScoreLayer(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.mlp = MLP_factory([[3 * args['enc_dim'], 1]] + args['layer_sizes'], layer_norm=args['layer_norm'])
        self.output_mlp = nn.Linear(args['enc_dim'], args['enc_dim'] + 1)

    def forward(self, target, graph, **unused):
        # target: (n_targets * sum(m_i), 1, d)
        # graph: (n_targets * sum(m_i), 2, d)

        enc_dim = target.shape[-1]
        target = target.reshape(-1, enc_dim)
        graph = graph.reshape(-1, 2 * enc_dim)

        encoding = self.mlp(torch.cat((target, graph), dim=-1))
        mlp_output = self.output_mlp(encoding)
        update = mlp_output[:, 1:]
        score = mlp_output[:, :1]

        scored_update = score * update

        return scored_update

class MLPConcatSeparateScoreLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.update_mlp = MLP_factory([[3 * args['enc_dim'], 1]] + args['layer_sizes'], layer_norm=args['layer_norm'])
        self.score_mlp = MLP_factory([[3 * args['enc_dim'], 1]] + args['score_layer_sizes'], layer_norm=args['layer_norm'])

    def forward(self, target, graph, **unused):
        # target: (n_targets * sum(m_i), 1, d)
        # graph: (n_targets * sum(m_i), 2, d)

        enc_dim = target.shape[-1]
        target = target.reshape(-1, enc_dim)
        graph = graph.reshape(-1, 2 * enc_dim)
        mlp_input = torch.cat((target, graph), dim=-1)
        update = self.update_mlp(mlp_input)
        score = self.score_mlp(mlp_input)

        scored_update = score * update

        return scored_update
