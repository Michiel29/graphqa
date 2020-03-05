import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel, roberta


import tasks

from models.encoder import encoder_dict
from models.gnn import gnn_dict


@register_model('encoder_gnn')
class EncoderGNNModel(BaseFairseqModel):

    def __init__(self, args, encoder, gnn_model):
        super().__init__()

        self.args = args

        self.encoder = encoder
        self.gnn_model = gnn_model

        mention_dim = args.mention_dim
        emb_dim = args.emb_dim

        # Define subgraph embedders
        self.embedder_dict = nn.ModuleDict()
        for arity in emb_dim.keys():
            self.embedder_dict[str(arity)] = nn.Linear(mention_dim, emb_dim[arity], padding_idx=0)

        # Define scoring MLPs
        self.mlp_dict = nn.ModuleDict()
        for k, v in args.layer_sizes.items():
            self.mlp_dict[k] = MLP_factory(v, dropout, layer_norm)

    def embed_subgraph(self, x):
        x_emb = {}
        for arity in x.keys():
            x_emb_idx = (x[arity] + 1).long()
            x_emb[arity] = self.embedder_dict[str(arity)](x_emb_idx)
        return x_emb

    def select_goal_encoding(x_enc, goal_arity):
        idx = (slice(None),) + tuple(range(goal_arity))
        goal_enc = x_enc[goal_arity][idx]
        return goal_enc

    def forward(self, batch):

        goal_arity = batch['goal_arity']

        text_encoding = self.encoder(batch['text'])

        # TODO: add subgraph selection model
        # subgraph = subgraph_selector(batch['goal_entities'], batch['mention'])

        subgraph_embedding = self.embed_subgraph(subgraph)

        subgraph_encoding = self.gnn_model(subgraph_embedding)

        goal_encoding = self.select_goal_encoding(subgraph_encoding, goal_arity)

        final_encoding = torch.cat((goal_encoding, text_encoding), dim=-1)

        score = self.mlp[goal_arity](final_encoding)
        normalized_scores = F.softmax(score, dim=-1)
        positive_scores = normalized_scores[Ellipsis, 0]

        return positive_scores


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        parser.add_argument('--gnn_type', type=str, default='nlm',
                            help='type of gnn model to use for inference')

    @classmethod
    def build_model(cls, args, task):

        encoder = encoder_dict[args.encoder_type](args)
        gnn_model = gnn_dict[args.gnn_type](args)

        return cls(args, encoder, gnn_model)

@register_model_architecture('encoder_gnn', 'encoder_gnn')
def encoder_gnn_architecture(args):
    pass
