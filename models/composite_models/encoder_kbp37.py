import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel, roberta
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture
from utils.diagnostic_utils import Diagnostic


@register_model('encoder_kbp37')
class EncoderKBP37Model(BaseFairseqModel):

    def __init__(self, args, encoder, task):
        super().__init__()

        self.args = args
        self.cls_mode = 'pytorch'

        self.encoder = encoder
        if args.use_pytorch_classifier:
            if args.encoder_output_layer_type in ['entity_start', 'entity_pooling_first_token', 'entity_pooling_first_token']:
                self.classifier = nn.Linear(2 * args.encoder_embed_dim, args.num_classes)
            elif args.encoder_output_layer_type in ['entity_start_linear']:
                self.classifier = nn.Linear(args.entity_dim, args.num_classes)
            else:
                self.classifier = nn.Linear(args.encoder_embed_dim, args.num_classes)

        self.task = task

    def forward(self, batch):

        text = batch['text'] # [batch_size, n_tokens]

        text_enc, _ = self.encoder(text, annotation=batch.get('annotation'))

        if self.cls_mode == 'sklearn':
            features = text_enc.cpu().detach().numpy()
            targets = batch['target'].cpu().detach().numpy()
            return features, targets
        else:
            scores = self.classifier(text_enc)

            # diag = Diagnostic(self.task.dictionary, self.task.entity_dictionary, self.task)
            # diag.inspect_batch(batch, scores=scores)

            return scores

    @classmethod
    def build_model(cls, args, task, encoder=None):
        if encoder is None:
            encoder = RobertaWrapper.build_model(args, task)
        return cls(args, encoder, task)

@register_model_architecture('encoder_kbp37', 'encoder_kbp37__roberta_small')
def roberta_small_architecture(args):
    small_architecture(args)

@register_model_architecture('encoder_kbp37', 'encoder_kbp37__roberta_base')
def roberta_base_architecture(args):
    base_architecture(args)

@register_model_architecture('encoder_kbp37', 'encoder_kbp37__roberta_large')
def roberta_large_architecture(args):
    large_architecture(args)
