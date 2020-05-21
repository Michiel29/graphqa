import math
from torch import nn
import torch
from modules.mlp import MLP_factory

class BaseLayer(nn.Module):
    def forward(self, candidate_rep, graph_rep, graph_sizes, put_indices, **unused):
        candidate_rep_repeat = torch.repeat_interleave(candidate_rep, graph_sizes, dim=0) # (sum(m_i), 1, d)
        candidate_update, graph_rep = self.update(candidate_rep_repeat, graph_rep, graph_sizes, put_indices, **unused)
        candidate_rep = candidate_rep.index_put(put_indices, candidate_update, accumulate=True)

        return candidate_rep, graph_rep

    def update(self, candidate_rep_repeat, graph_rep, graph_sizes, put_indices, **unused):
        raise NotImplementedError


class MLPConcatLayer(BaseLayer):

    def __init__(self, args):
        super().__init__()
        self.mlp = MLP_factory([[3 * args['enc_dim'], 1]] + args['layer_sizes'], layer_norm=args['layer_norm'])

    def update(self, candidate, graph, **unused):
        # target: (n_targets * sum(m_i), d)
        # graph: (n_targets * sum(m_i), 2, d)

        enc_dim = candidate.shape[-1]
        graph_reshape = graph.reshape(-1, 2 * enc_dim)
        candidate_output = self.mlp(torch.cat((candidate, graph_reshape), dim=-1))

        return candidate_output, graph


class MLPLinearConcatLayer(BaseLayer):

    def __init__(self, args):
        super().__init__()
        self.graph_linear = nn.Linear(2*args['enc_dim'], args['linear_dim'])
        self.mlp = MLP_factory([[2 * args['enc_dim'], 1]] + args['layer_sizes'], layer_norm=args['layer_norm'])

    def update(self, candidate, graph, **unused):
        # target: (n_targets * sum(m_i), d)
        # graph: (n_targets * sum(m_i), 2, d)

        enc_dim = candidate.shape[-1]
        graph_reshape = graph.reshape(-1, 2*enc_dim)
        graph_emb = self.graph_linear(graph_reshape)
        candidate_output = self.mlp(torch.cat((candidate, graph_emb), dim=-1))

        return candidate_output, graph


class MLPConcatScoreLayer(BaseLayer):

    def __init__(self, args):
        super().__init__()

        self.mlp = MLP_factory([[3 * args['enc_dim'], 1]] + args['layer_sizes'], layer_norm=args['layer_norm'])
        self.output_mlp = nn.Linear(args['enc_dim'], args['enc_dim'] + 1)

    def update(self, candidate, graph, **unused):
        # target: (n_targets * sum(m_i), d)
        # graph: (n_targets * sum(m_i), 2, d)

        enc_dim = candidate.shape[-1]
        graph_reshape = graph.reshape(-1, 2 * enc_dim)

        encoding = self.mlp(torch.cat((candidate, graph_reshape), dim=-1))
        mlp_output = self.output_mlp(encoding)
        update = mlp_output[:, 1:]
        score = mlp_output[:, :1]

        scored_update = score * update

        return scored_update, graph

class MLPConcatSeparateScoreLayer(BaseLayer):

    def __init__(self, args):
        super().__init__()
        self.update_mlp = MLP_factory([[3 * args['enc_dim'], 1]] + args['layer_sizes'], layer_norm=args['layer_norm'])
        self.score_mlp = MLP_factory([[3 * args['enc_dim'], 1]] + args['score_layer_sizes'], layer_norm=args['layer_norm'])

    def update(self, candidate, graph, **unused):
        # target: (n_targets * sum(m_i), d)
        # graph: (n_targets * sum(m_i), 2, d)

        enc_dim = candidate.shape[-1]
        graph_reshape = graph.reshape(-1, 2 * enc_dim)
        mlp_input = torch.cat((candidate, graph_reshape), dim=-1)
        update = self.update_mlp(mlp_input)
        score = self.score_mlp(mlp_input)

        scored_update = score * update

        return scored_update, graph

class MLPConcatMutualLayer(BaseLayer):
    def __init__(self, args):
        super().__init__()
        self.mlp = MLP_factory([[3 * args['enc_dim'], 1]] + args['layer_sizes'], layer_norm=args['layer_norm'])

    def update(self, candidate_input, graph, graph_sizes, put_indices, **unused):
        # target: (n_targets * sum(m_i), d)
        # graph: (n_targets * sum(m_i), 2, d)

        enc_dim = candidate_input.shape[-1]
        graph_reshape = graph.reshape(-1, 2 * enc_dim)
        mlp_output = self.mlp(torch.cat((candidate_input, graph_reshape), dim=-1))
        candidate_output = mlp_output[:, :enc_dim]
        graph_output = mlp_output[:, enc_dim:].reshape(-1, 2, enc_dim)
        return candidate_output, graph_output



class AttentionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        enc_dim = args['enc_dim']
        self.heads = args['heads']
        self.head_dim = enc_dim // self.heads
        dropout = args['dropout']

        self.q_linear = nn.Linear(enc_dim, enc_dim)
        self.v_linear = nn.Linear(2 * enc_dim, enc_dim)
        self.k_linear = nn.Linear(2 * enc_dim, enc_dim)
        self.attention_linear = nn.Linear(enc_dim, enc_dim)

        self.mlp = MLP_factory([[enc_dim, 2]])

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(enc_dim)
        self.norm_2 = nn.LayerNorm(enc_dim)

    def forward(self, candidate_input, graph, graph_sizes, put_indices, **unused):
        candidate_rep = torch.repeat_interleave(candidate_input, graph_sizes, dim=0) # (sum(m_i), 1, d)
        enc_dim = candidate_rep.shape[-1]
        graph = graph.reshape(-1, 2 * enc_dim)
        device = candidate_rep.device

        candidate_q = self.q_linear(candidate_rep).view(-1, self.heads, self.head_dim) # (n_targets * sum(m_i), h, d_h)
        graph_v = self.v_linear(graph).view(-1, self.heads, self.head_dim) # (n_targets * sum(m_i), h, d_h)
        graph_k = self.k_linear(graph).view(-1, self.heads, self.head_dim) # (n_targets * sum(m_i), h, d_h)

        scores = (candidate_q * graph_k).sum(dim=-1) / math.sqrt(self.head_dim)
        exp_scores = scores.exp()
        score_denominator = torch.zeros((len(graph_sizes), self.heads), device=device)
        score_denominator = score_denominator.index_put(put_indices, exp_scores, accumulate=True)
        score_denominator = score_denominator.repeat_interleave(graph_sizes, dim=0)
        normalized_scores = exp_scores / score_denominator

        head_attention_output = normalized_scores.unsqueeze(-1) * graph_v
        attention_output = head_attention_output.contiguous().view(-1, enc_dim)
        attention_output = self.attention_linear(attention_output)
        attention_output = self.dropout_1(attention_output)
        candidate_output = candidate_input.index_put(put_indices, attention_output, accumulate=True)
        candidate_output = self.norm_1(candidate_output)
        candidate_output = self.mlp(candidate_output)
        candidate_output = self.dropout_2(candidate_output)
        candidate_output = self.norm_2(candidate_output)

        return candidate_output, graph
