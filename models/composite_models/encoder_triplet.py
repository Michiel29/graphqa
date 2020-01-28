import torch.nn as nn
import torch.nn.functional as F

from fairseq import models
from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel, roberta


from models.encoder import encoder_dict
from models.triplet import triplet_dict

@register_model('encoder_triplet')
class EncoderTripletModel(BaseFairseqModel):

    def __init__(self, args, encoder, triplet_model):
        super().__init__()

        self.args = args

        self.encoder = encoder
        self.ent_emb = nn.Embedding(args.n_entities, args.entity_dim)
        self.triplet_model = triplet_model

    def forward(self, batch):

        mention_encoding = self.encoder(batch['mention'])

        head_emb = self.emb(batch['head'])
        tail_emb = self.emb(batch['tail'])

        self.score = self.triplet_model(mention_encoding, head_emb, tail_emb)

        score = self.triplet_model(batch['head'], mention_encoding, batch['tail'])
        normalized_scores = F.softmax(score, dim=-1)
        positive_scores = normalized_scores[Ellipsis, 0]

        return positive_scores


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        parser.add_argument('--triplet_type', type='string', default='distmult',
                            help='type of triplet model to use for inference')

    @classmethod
    def build_model(cls, args, task):
        
        encoder = encoder_dict[args.encoder_type](args)
        triplet_model = triplet_dict[args.triplet_type](args)

        return cls(args, encoder, triplet_model)

@register_model_architecture('encoder_triplet', 'encoder_triplet')
def encoder_triplet_architecture(args):
    pass