from torch import nn
from torch.nn import functional as F
import torch

from modules.mlp import MLP_factory

class BoW(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        weights = torch.ones(len(dictionary))
        weights[dictionary.special_tokens()] = 0
        self.weights = nn.Parameter(weights)

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]
        mask = self.weights[src_tokens]
        avg_emb = torch.bmm(mask.unsqueeze(-2), x).squeeze(-2) / torch.sum(mask, dim=-1, keepdim=True)
        return avg_emb


class BoWLinear(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        weights = torch.ones(len(dictionary))
        weights[dictionary.special_tokens()] = 0
        self.weights = nn.Parameter(weights)
        self.linear = nn.Linear(args.encoder_embed_dim, args.encoder_representation_dim)

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]
        mask = self.weights[src_tokens]
        avg_emb = torch.bmm(mask.unsqueeze(-2), x).squeeze(-2) / torch.sum(mask, dim=-1, keepdim=True)
        linear_projection = self.linear(avg_emb)
        return linear_projection


class EntityStart(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        if args.mask_type == 'head_tail':
            self.head_idx = dictionary.head()
            self.tail_idx = dictionary.tail()
        elif args.mask_type == 'start_end':
            self.head_idx = dictionary.e1_start()
            self.tail_idx = dictionary.e2_start()
        else:
            raise Exception('EntityStart is unsupported for the mask type %s' % str(args.mask_type))

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]

        head_mask = (src_tokens == self.head_idx) # [batch_size, length]
        tail_mask = (src_tokens == self.tail_idx) # [batch_size, length]

        head_sum = torch.bmm(head_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]

        head_tail_concat = torch.cat((head_sum, tail_sum), dim=-1) # [batch_size, 2 * enc_dim]

        return head_tail_concat


class EntityStartLinear(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        if args.mask_type == 'head_tail':
            self.head_idx = dictionary.head()
            self.tail_idx = dictionary.tail()
        elif args.mask_type == 'start_end':
            self.head_idx = dictionary.e1_start()
            self.tail_idx = dictionary.e2_start()
        else:
            raise Exception('EntityStartLinear is unsupported for the mask type %s' % str(args.mask_type))

        self.linear = nn.Linear(2 * args.encoder_embed_dim, args.entity_dim)

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]

        head_mask = (src_tokens == self.head_idx) # [batch_size, length]
        tail_mask = (src_tokens == self.tail_idx) # [batch_size, length]

        head_sum = torch.bmm(head_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]

        head_tail_concat = torch.cat((head_sum, tail_sum), dim=-1) # [batch_size, 2 * enc_dim]

        linear_projection = self.linear(head_tail_concat)

        return linear_projection


class EntityTargetLinear(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        if args.mask_type == 'head_tail':
            self.head_idx = dictionary.head()
            self.tail_idx = dictionary.tail()
        elif args.mask_type == 'start_end':
            self.head_idx = dictionary.e1_start()
            self.tail_idx = dictionary.e2_start()
        else:
            raise Exception('EntityTarget is unsupported for the mask type %s' % str(args.mask_type))

        self.linear = nn.Linear(args.encoder_embed_dim, args.entity_dim)

    def forward(self, x, src_tokens, replace_heads=None, **unused):
        # x: [batch_size, length, enc_dim]

        head_mask = (src_tokens == self.head_idx) # [batch_size, length]
        tail_mask = (src_tokens == self.tail_idx) # [batch_size, length]

        head_sum = torch.bmm(head_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]

        if replace_heads is not None:
            head_or_tail = replace_heads * head_sum + (1 - replace_heads) * tail_sum
            return self.linear(head_or_tail)
        else:
            head_sum = self.linear(head_sum)
            tail_sum = self.linear(tail_sum)
            return torch.cat((head_sum, tail_sum), dim=-1) # [batch_size, 2 * enc_dim]


class EntitySplitLinear(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        if args.mask_type == 'head_tail':
            self.head_idx = dictionary.head()
            self.tail_idx = dictionary.tail()
        elif args.mask_type == 'start_end':
            self.head_idx = dictionary.e1_start()
            self.tail_idx = dictionary.e2_start()
        else:
            raise Exception('EntitySplitLinear is unsupported for the mask type %s' % str(args.mask_type))

        self.linear = nn.Linear(args.encoder_embed_dim, args.entity_dim)

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]
        head_mask = (src_tokens == self.head_idx) # [batch_size, length]
        tail_mask = (src_tokens == self.tail_idx) # [batch_size, length]

        head_sum = torch.bmm(head_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True)
        head_sum = self.linear(head_sum)
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True)
        tail_sum = self.linear(tail_sum)
        return torch.cat((head_sum, tail_sum), dim=-1)


class EntityStartLayerNorm(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        if args.mask_type == 'head_tail':
            self.head_idx = dictionary.head()
            self.tail_idx = dictionary.tail()
        elif args.mask_type == 'start_end':
            self.head_idx = dictionary.e1_start()
            self.tail_idx = dictionary.e2_start()
        else:
            raise Exception('EntityStartLayerNorm is unsupported for the mask type %s' % str(args.mask_type))

        self.layer_norm = nn.LayerNorm(2*args.encoder_embed_dim)

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]

        head_mask = (src_tokens == self.head_idx) # [batch_size, length]
        tail_mask = (src_tokens == self.tail_idx) # [batch_size, length]

        head_sum = torch.bmm(head_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]

        head_tail_concat = torch.cat((head_sum, tail_sum), dim=-1) # [batch_size, 2 * enc_dim]

        output = self.layer_norm(head_tail_concat)

        return output


class EntityStartMLP(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        if args.mask_type == 'head_tail':
            self.head_idx = dictionary.head()
            self.tail_idx = dictionary.tail()
        elif args.mask_type == 'start_end':
            self.head_idx = dictionary.e1_start()
            self.tail_idx = dictionary.e2_start()
        else:
            raise Exception('EntityStartMLP is unsupported for the mask type %s' % str(args.mask_type))

        layer_sizes = [[2 * args.encoder_embed_dim, 1], [args.mlp_args['n_hidden_dim'], args.mlp_args['n_hidden_layers']], [args.entity_dim, 1]]
        self.mlp = MLP_factory(layer_sizes, dropout=args.mlp_args['dropout'], layer_norm=args.mlp_args['layer_norm'])

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]

        head_mask = (src_tokens == self.head_idx) # [batch_size, length]
        tail_mask = (src_tokens == self.tail_idx) # [batch_size, length]

        head_sum = torch.bmm(head_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).type_as(x), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]

        head_tail_concat = torch.cat((head_sum, tail_sum), dim=-1) # [batch_size, 2 * enc_dim]

        x = self.mlp(head_tail_concat)

        return x


class EntityPoolingFirstToken(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()

    def forward(self, x, src_tokens, annotation, **unused):
        # x: [batch_size, length, enc_dim]
        head_first_tokens = torch.max(annotation == 0, dim=1)[1]
        tail_first_tokens = torch.max(annotation == 1, dim=1)[1]
        arange = torch.arange(x.shape[0], device=x.device)

        head_tail_concat = torch.cat(
            (
                x[arange, head_first_tokens],
                x[arange, tail_first_tokens],
            ),
            dim=-1,
        ) # [batch_size, 2 * enc_dim]
        return head_tail_concat



class EntityConcat(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()

    def forward(self, x, src_tokens, annotation, **unused):
        # x: [batch_size, length, enc_dim]

        entity_rep = x.gather(1, annotation.view(-1, 4, 1).expand(-1, -1, x.shape[-1])) # [batch_size, 4, enc_dim]
        head_tail_concat = entity_rep.view(x.shape[0], -1) # [batch_size, enc_dim * 4]

        return head_tail_concat


class EntityConcatLinear(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.linear = nn.Linear(4 * args.encoder_embed_dim, args.entity_dim)

    def forward(self, x, src_tokens, annotation, **unused):
        # x: [batch_size, length, enc_dim]

        entity_rep = x.gather(1, annotation.view(-1, 4, 1).expand(-1, -1, x.shape[-1])) # [batch_size, 4, enc_dim]
        head_tail_concat = entity_rep.view(x.shape[0], -1) # [batch_size, enc_dim * 4]
        relation_representation = self.linear(head_tail_concat)

        return relation_representation

class MentionConcatLinear(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.linear = nn.Linear(2 * args.encoder_embed_dim, args.entity_dim)

    def forward(self, x, src_tokens, annotation, **unused):
        # x: [batch_size, length, enc_dim]


        mention_representation = x.gather(1, annotation.view(-1, 2, 1).expand(-1, -1, x.shape[-1])) # [batch_size, 2, enc_dim]
        mention_representation = mention_representation.view(x.shape[0], -1) # [batch_size, enc_dim * 2]
        mention_representation = self.linear(mention_representation) # [batch_size, entity_dim]

        return mention_representation

class MentionConcatMLP(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        head_args = args.head_args
        self.projection_mlp = MLP_factory([[2 * args.encoder_embed_dim, 1]] + head_args['layer_sizes'] + [[args.entity_dim, 1]], layer_norm=head_args['layer_norm'], dropout=head_args['dropout'])

    def forward(self, x, src_tokens, annotation, **unused):
        # x: [batch_size, length, enc_dim]


        mention_representation = x.gather(1, annotation.view(-1, 2, 1).expand(-1, -1, x.shape[-1])) # [batch_size, 2, enc_dim]
        mention_representation = mention_representation.view(x.shape[0], -1) # [batch_size, enc_dim * 2]
        mention_representation = self.projection_mlp(mention_representation) # [batch_size, entity_dim]

        return mention_representation


class RelationAttentionLayer(nn.Module):
    def __init__(self, encoder_embed_dim, n_heads, dropout, ffn_dim):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = self.encoder_embed_dim // n_heads
        self.linear_relation_query = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)
        self.linear_relation_value = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)
        self.linear_relation_key = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)
        self.linear_token_key = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)
        self.linear_token_value = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)

        self.linear_out = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)

        self.attention_layer_norm = nn.LayerNorm(self.encoder_embed_dim)
        self.fc1 = nn.Linear(self.encoder_embed_dim, ffn_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(ffn_dim, self.encoder_embed_dim)
        self.connected_layer_norm = nn.LayerNorm(self.encoder_embed_dim)

    def forward(self, relation_representation, token_representation):
        n_tokens = token_representation.shape[1]
        relation_query = self.linear_relation_query(relation_representation).view(-1, 1, self.n_heads, self.head_dim) # (batch_size, 1, n_heads, encoder_dim / n_heads)
        relation_value = self.linear_relation_value(relation_representation).view(-1, self.n_heads, self.head_dim) # (batch_size, 1, n_heads, encoder_dim / n_heads)
        relation_key = self.linear_relation_key(relation_representation).view(-1, 1, self.n_heads, self.head_dim) # (batch_size, 1, n_heads, encoder_dim / n_heads)
        token_key = self.linear_token_key(token_representation).view(-1, n_tokens, self.n_heads, self.head_dim) # (batch_size, n_tokens, n_heads, encoder_dim / n_heads)
        token_value = self.linear_token_value(token_representation).view(-1, n_tokens, self.n_heads, self.head_dim) # (batch_size, n_tokens, n_heads, encoder_dim / n_heads)

        self_attention_score = (relation_query * relation_key).squeeze(1).sum(axis=-1)
        token_attention_scores = (relation_query * token_key).sum(axis=-1)
        score_denominator = self_attention_score + token_attention_scores.sum(axis=1)
        self_attention_score_normalized = self_attention_score / score_denominator
        token_attention_score_normalized = token_attention_scores / score_denominator.unsqueeze(-2)

        update = (self_attention_score_normalized.unsqueeze(-1) * relation_value) + (token_attention_score_normalized.unsqueeze(-1) * token_value).sum(axis=1)
        update = self.linear_out(update.view(-1, self.encoder_embed_dim))
        update = F.dropout(update, self.dropout, self.training)

        new_relation_representation = relation_representation + update
        new_relation_representation = self.attention_layer_norm(new_relation_representation)

        update = self.activation(self.fc1(new_relation_representation))
        update = F.dropout(update, self.dropout, self.training)
        update = self.fc2(update)
        new_relation_representation = new_relation_representation + update
        new_relation_representation = self.connected_layer_norm(new_relation_representation)

        return new_relation_representation

class EntityConcatAttention(nn.Module):
    """ Generate relation representation for mentions i, j by concatenating first and last token of the mentions, then applying attention layer. More specifically,

        M_i = [h(s_i); h(e_i)] where h(s_i) and h(e_i) are representations of starting and ending token of mention, respectively
        R_i,j = g([M_i; M_j]) where g applies an attention block with M_i; M_j as the query and M_i;M_j and the token representations in the passage as keys and values
    """

    def __init__(self, args, dictionary):
        super().__init__()
        self.linear_projection = nn.Linear(4 * args.encoder_embed_dim, args.encoder_embed_dim)
        self.relation_layers = nn.ModuleList()
        for layer_idx in range(args.n_relation_layers):
            self.relation_layers.append(RelationAttentionLayer(args.encoder_embed_dim, args.encoder_attention_heads, args.dropout, args.encoder_ffn_embed_dim))

    def forward(self, x, src_tokens, annotation, **unused):
        # x: [batch_size, length, enc_dim]

        entity_rep = x.gather(1, annotation.view(-1, 4, 1).expand(-1, -1, x.shape[-1])) # [batch_size, 4, enc_dim]
        head_tail_concat = entity_rep.view(x.shape[0], -1) # [batch_size, enc_dim * 4]
        relation_representation = self.linear_projection(head_tail_concat)
        for layer in self.relation_layers:
            relation_representation = layer(relation_representation, x)
        return relation_representation

class AllRelation(nn.Module):
    """
    As EntityConcatAttention, but for all pairs of mentions in passage"""

    def __init__(self, args, dictionary):
        super().__init__()
        head_args = args.head_args
        self.relation_mlp = MLP_factory([(args.encoder_embed_dim * 4, 1)] + head_args['layer_sizes'] + [(2 * args.entity_dim, 1)], dropout=head_args['dropout'], layer_norm=head_args['layer_norm'])
        self.score_mlp = MLP_factory([(args.encoder_embed_dim * 4, 1)] + head_args['layer_sizes'] + [(1, 1)], dropout=head_args['dropout'], layer_norm=head_args['layer_norm'])

    def forward(self, x, src_tokens, mask_annotation, all_annotations, n_annotations, relation_entity_indices_left, relation_entity_indices_right, **unused):
        # x: [batch_size, length, enc_dim]
        batch_size = x.shape[0]
        encoder_embed_dim = x.shape[-1]
        idx = torch.arange(batch_size, device=x.device)
        idx_repeated = idx.repeat_interleave(2 * n_annotations)
        entity_representation = x[idx_repeated, all_annotations.reshape(1,-1)].reshape(-1, 2 * encoder_embed_dim)
        entity_representation_repeated_left = entity_representation[relation_entity_indices_left]
        entity_representation_repeated_right = entity_representation[relation_entity_indices_right]
        relation_input = torch.cat((entity_representation_repeated_left, entity_representation_repeated_right), dim=-1)
        relation_representation = self.relation_mlp(relation_input)
        relation_score = torch.abs(self.score_mlp(relation_input).squeeze())
        return entity_representation, relation_representation, relation_score

class CLSTokenLinear(nn.Module):
    def __init__(self, args, dictionary):
        super().__init__()
        self.linear = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)

    def forward(self, x, src_tokens, **unused):
        return self.linear(x[:, 0, :])


class CLSTokenLayerNorm(nn.Module):
    def __init__(self, args, dictionary):
        super().__init__()
        self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)

    def forward(self, x, src_tokens, **unused):
        return self.layer_norm(x[:, 0, :])


encoder_head_dict = {
    'bag_of_words': BoW,
    'bag_of_words_linear': BoWLinear,
    'entity_start': EntityStart,
    'entity_start_linear': EntityStartLinear,
    'entity_start_mlp': EntityStartMLP,
    'entity_start_layer_norm': EntityStartLayerNorm,
    'entity_split_linear': EntitySplitLinear,
    'entity_target_linear': EntityTargetLinear,
    'entity_pooling_first_token': EntityPoolingFirstToken,
    'entity_concat': EntityConcat,
    'entity_concat_linear': EntityConcatLinear,
    'mention_concat_linear': MentionConcatLinear,
    'mention_concat_mlp': MentionConcatMLP,
    'entity_concat_attention': EntityConcatAttention,
    'all_relation': AllRelation,
    'cls_token_linear': CLSTokenLinear,
    'cls_token_layer_norm': CLSTokenLayerNorm,
}
