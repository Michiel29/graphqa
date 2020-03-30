from torch import nn
import torch


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

        head_sum = torch.bmm(head_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]

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

        head_sum = torch.bmm(head_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]

        head_tail_concat = torch.cat((head_sum, tail_sum), dim=-1) # [batch_size, 2 * enc_dim]

        linear_projection = self.linear(head_tail_concat)

        return linear_projection

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
            raise Exception('EntityStartLinear is unsupported for the mask type %s' % str(args.mask_type))

        self.layer_norm = nn.LayerNorm(2*args.encoder_embed_dim)

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]

        head_mask = (src_tokens == self.head_idx) # [batch_size, length]
        tail_mask = (src_tokens == self.tail_idx) # [batch_size, length]

        head_sum = torch.bmm(head_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]

        head_tail_concat = torch.cat((head_sum, tail_sum), dim=-1) # [batch_size, 2 * enc_dim]

        output = self.layer_norm(head_tail_concat)

        return output

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
    'entity_start_layer_norm': EntityStartLayerNorm,
    'entity_pooling_first_token': EntityPoolingFirstToken,
    'cls_token_linear': CLSTokenLinear,
    'cls_token_layer_norm': CLSTokenLayerNorm,
}
