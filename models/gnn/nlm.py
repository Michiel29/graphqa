import torch
from torch import nn
import torch.nn.functional as F
from mlp import MLP_factory
from itertools import combinations_with_replacement, product, permutations

class NLMEncoder(ModuleClass):
    arg_names = ['n_encoder_layers', 'first_layer_sizes', 'layer_sizes', 'n_entities']

    def __init__(self, n_encoder_layers, first_layer_sizes, layer_sizes, n_entities):
        super().__init__()

        self.first_layer = NLMEncoderLayer(first_layer_sizes, n_entities)
        self.layers = nn.ModuleList([NLMEncoderLayer(layer_sizes, n_entities) for i in range(n_encoder_layers-1)])

    def forward(self, x): 
        x = self.first_layer(x)
        for layer in self.layers:
            x = layer(x)
    
        return x

class NLMEncoderLayer(nn.Module):
    def __init__(self, layer_sizes, n_entities):
        super().__init__()

        self.n_entities = n_entities

        # Create list of arities
        self.arities = [int(a) for a in list(layer_sizes.keys())]
        self.min_arity = min(self.arities) 
        self.max_arity = max(self.arities) 

        # Create MLP dict
        self.mlp_dict = nn.ModuleDict() 
        for k, v in layer_sizes.items():
            self.mlp_dict[k] = MLP_factory(v)

        # Generate predicate permuation indices
        self.predicate_perm_idx = {}
        for a in range(self.min_arity, self.max_arity+1):
            self.predicate_perm_idx[a] = self.permute(a, n_entities)

    def permute(self, arity, n_entities):
        predicate_args = list(product(range(n_entities), repeat=arity)) 
        predicate_args = [list(p) for p in predicate_args]

        predicate_args_perm = [list(permutations(p)) for p in predicate_args]
        predicate_args_perm = torch.LongTensor(predicate_args_perm).cuda()
        predicate_args_perm = predicate_args_perm.reshape(-1, arity)

        for a in range(arity):
            if a == 0:
                predicate_perm_idx = (n_entities ** a) * predicate_args_perm[:, a]
            else:
                predicate_perm_idx += (n_entities ** a) * predicate_args_perm[:, a]

        return predicate_perm_idx

    def expand(self, emb, arity):
        emb_shape = list(emb.shape)
        n_entities = emb_shape[1]
        expand_idx = [-1] * len(emb_shape)
        expand_idx.insert(-1, n_entities)
        emb_expand = emb.unsqueeze(-2).expand(expand_idx)
        return emb_expand

    def reduce(self, emb):
        emb_reduce_max, _ = torch.max(emb, dim=-2)
        emb_reduce_min, _ = torch.min(emb, dim=-2)
        return emb_reduce_max, emb_reduce_min

    def forward(self, embeddings):

        # Initialize embedding dicts
        embeddings_exp = {}  
        embeddings_red = {}
        embeddings_cat = {}
        embeddings_perm = {}
        embeddings_new = {}

        # Perform expansion step
        for a in self.arities:
            if a == self.min_arity:
                embeddings_exp[a] = torch.Tensor([]).cuda()
            else:
                embeddings_exp[a] = self.expand(embeddings[a-1], a)

        # Perform reduction step
        for a in self.arities:
            if a == self.max_arity:
                embeddings_red[a] = [torch.Tensor([]).cuda()]
            else:
                embeddings_red[a] = list(self.reduce(embeddings[self.max_arity-a+1]))

        # Perform inter-group computation
        for a in self.arities:
            cur_emb_exp = embeddings_exp[a]
            cur_emb_red = torch.cat(embeddings_red[a], dim=-1)
            embeddings_cat[a] = torch.cat((cur_emb_exp, embeddings[a], cur_emb_red), dim=-1)

        # Perform intra-group computation (permutation step)
        batch_size = next(iter(embeddings.values())).shape[0]
        for a in self.arities:
            if a == 1:
                embeddings_perm[a] = embeddings_cat[a]
            else:
                cur_emb_cat = embeddings_cat[a].reshape(batch_size, -1, embeddings_cat[a].shape[-1])
                cur_shape = [batch_size] + [self.n_entities]*(a) + [-1]
                cur_emb_perm = torch.index_select(cur_emb_cat, 1, self.predicate_perm_idx[a]).reshape(cur_shape)
                embeddings_perm[a] = cur_emb_perm

        # Perform intra-group computation (MLP step)
        for a in self.arities:
            embeddings_new[a] = self.mlp_dict[str(a)](embeddings_perm[a])


        return embeddings_new
