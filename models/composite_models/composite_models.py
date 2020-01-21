from pudb import set_trace

import torch.nn as nn

from fairseq.models import register_model
from fairseq.models import BaseFairseqModel, FairseqLanguageModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from models.encoder.roberta import RobertaEncoder, base_architecture
from models.inference.triplet import triplet_dict

@register_model('roberta_triplet')
class RobertaTripletModel(FairseqLanguageModel):
#class RobertaTripletModel(BaseFairseqModel):

    def __init__(self, args, decoder, triplet):
        super().__init__(decoder)

        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.decoder = decoder
        self.triplet = triplet
        self.classification_heads = nn.ModuleDict()
    
    @classmethod
    def build_model(cls, args, task):

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        if not hasattr(args, 'triplet_type'):
            args.triplet_type = 'distmult'

        encoder = RobertaEncoder(args, task.source_dictionary)
        
        triplet = triplet_dict[args.triplet_type]()

        return cls(args, encoder, triplet)


model_dict = {'roberta_triplet': RobertaTripletModel}
