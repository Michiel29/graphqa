from torch import nn
import torch


class BoW(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.head_idx = dictionary.head()
        self.tail_idx = dictionary.tail()

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]
        mask = ((src_tokens != self.padding_idx) & (src_tokens != self.head_idx) & (src_tokens != self.tail_idx)).unsqueeze(-1) # [batch_size, length, enc_dim]
        masked_emb = x * mask
        mask_sum = mask.sum(dim=-2) # [batch_size, enc_dim]
        avg_emb = torch.sum(masked_emb, dim=-2) / mask_sum

        return avg_emb


class BoWLinear(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.head_idx = dictionary.head()
        self.tail_idx = dictionary.tail()

        self.linear = nn.Linear(args.encoder_embed_dim, args.encoder_representation_dim)

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]
        mask = ((src_tokens != self.padding_idx) & (src_tokens != self.head_idx) & (src_tokens != self.tail_idx)).unsqueeze(-1) # [batch_size, length, enc_dim]
        masked_emb = x * mask
        mask_sum = mask.sum(dim=-2) # [batch_size, enc_dim]
        avg_emb = torch.sum(masked_emb, dim=-2) / mask_sum

        linear_projection = self.linear(avg_emb)

        return linear_projection


class HeadTailConcat(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        if args.mask_type == 'head_tail':
            self.head_idx = dictionary.head()
            self.tail_idx = dictionary.tail()
        elif args.mask_type == 'start_end':
            self.head_idx = dictionary.e1_start()
            self.tail_idx = dictionary.e2_start()
        else:
            raise Exception('HeadTailConcat is unsupported for the mask type %s' % str(args.mask_type))

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]

        head_mask = (src_tokens == self.head_idx) # [batch_size, length]
        tail_mask = (src_tokens == self.tail_idx) # [batch_size, length]

        head_sum = torch.bmm(head_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]

        head_tail_concat = torch.cat((head_sum, tail_sum), dim=-1) # [batch_size, 2 * enc_dim]

        return head_tail_concat


class HeadTailConcatLinear(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        if args.mask_type == 'head_tail':
            self.head_idx = dictionary.head()
            self.tail_idx = dictionary.tail()
        elif args.mask_type == 'start_end':
            self.head_idx = dictionary.e1_start()
            self.tail_idx = dictionary.e2_start()
        else:
            raise Exception('HeadTailConcatLinear is unsupported for the mask type %s' % str(args.mask_type))

        self.linear = nn.Linear(2*args.encoder_embed_dim, args.encoder_representation_dim)

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]

        head_mask = (src_tokens == self.head_idx) # [batch_size, length]
        tail_mask = (src_tokens == self.tail_idx) # [batch_size, length]

        head_sum = torch.bmm(head_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(head_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]
        tail_sum = torch.bmm(tail_mask.unsqueeze(-2).float(), x).squeeze(-2) / torch.sum(tail_mask, dim=-1, keepdim=True) # [batch_size, enc_dim]

        head_tail_concat = torch.cat((head_sum, tail_sum), dim=-1) # [batch_size, 2 * enc_dim]

        linear_projection = self.linear(head_tail_concat)

        return linear_projection


class EntityStart(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.e1_start_idx = dictionary.e1_start()
        self.e2_start_idx = dictionary.e2_start()

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]

        mask_e1 = (src_tokens == self.e1_start_idx).unsqueeze(-1) # [batch_size, length, 1]
        mask_e2 = (src_tokens == self.e2_start_idx).unsqueeze(-1) # [batch_size, length, 1]

        emb_e1 = torch.sum(x * mask_e1, dim=-2) # [batch_size, enc_dim]
        emb_e2 = torch.sum(x * mask_e2, dim=-2) # [batch_size, enc_dim]

        e1_e2_concat = torch.cat((emb_e1, emb_e2), dim=-1) # [batch_size, 2 * enc_dim]

        return e1_e2_concat


class EntityStartLinear(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.e1_start_idx = dictionary.e1_start()
        self.e2_start_idx = dictionary.e2_start()

        self.linear = nn.Linear(2*args.encoder_embed_dim, args.encoder_representation_dim)

    def forward(self, x, src_tokens, **unused):
        # x: [batch_size, length, enc_dim]

        mask_e1 = (src_tokens == self.e1_start_idx).unsqueeze(-1) # [batch_size, length, 1]
        mask_e2 = (src_tokens == self.e2_start_idx).unsqueeze(-1) # [batch_size, length, 1]

        emb_e1 = torch.sum(x * mask_e1, dim=-2) # [batch_size, enc_dim]
        emb_e2 = torch.sum(x * mask_e2, dim=-2) # [batch_size, enc_dim]

        e1_e2_concat = torch.cat((emb_e1, emb_e2), dim=-1) # [batch_size, 2 * enc_dim]

        linear_projection = self.linear(e1_e2_concat)

        return e1_e2_concat


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
    'head_tail_concat': HeadTailConcat,
    'head_tail_concat_linear': HeadTailConcatLinear,
    'entity_start': EntityStart,
    'entity_start_linear': EntityStartLinear,
    'cls_token_linear': CLSTokenLinear,
    'cls_token_layer_norm': CLSTokenLayerNorm,
}
