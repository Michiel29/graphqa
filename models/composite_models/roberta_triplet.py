import torch.nn as nn

from fairseq.models import register_model
from fairseq.models import BaseFairseqModel, FairseqLanguageModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from models.encoder.roberta import RobertaEncoder, base_architecture
from models.triplet import TripletModel

@register_model('roberta_triplet')
class RobertaTripletModel(FairseqLanguageModel):
#class RobertaTripletModel(BaseFairseqModel):

    def __init__(self, args, decoder, triplet_model):
        super().__init__(decoder)

        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.decoder = decoder
        self.triplet = triplet_model
        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')

    @classmethod
    def build_model(cls, args, task):

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)

        triplet_type = getattr(args, 'triplet_type', 'distmult')
        triplet_model = TripletModel(triplet_type, args.n_entities, args.entity_dim, args)

        return cls(args, encoder, triplet_model)

