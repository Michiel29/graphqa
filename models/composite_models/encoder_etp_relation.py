import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel

import tasks
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture


@register_model('encoder_etp_relation')
class EncoderETPRelation(BaseFairseqModel):

    def __init__(self, args, encoder, n_entities):
        super().__init__()

        self.args = args

        self.entity_dim = args.entity_dim
        self.encoder_embed_dim = args.encoder_embed_dim

        self.encoder = encoder
        self.entity_embedder = nn.Embedding(n_entities, args.entity_dim)

    def forward(self, batch):

        n_annotations = batch['n_annotations']
        batch_size = len(n_annotations)
        device = n_annotations.device

        relation_entity_indices_left = batch['relation_entity_indices_left']
        relation_entity_indices_right = batch['relation_entity_indices_right']

        entity_replacements = batch['entity_replacements']
        n_negatives = entity_replacements.shape[1]

        encoder_output, _ = self.encoder(batch['text'], mask_annotation=batch['mask_annotation'], all_annotations=batch['all_annotations'], n_annotations=n_annotations, relation_entity_indices_left=relation_entity_indices_left, relation_entity_indices_right=relation_entity_indices_right)
        query_entity_representation, query_relation_representation, query_relation_scores = encoder_output

        float_type = query_entity_representation.dtype
        selected_relation_indices = torch.arange(len(relation_entity_indices_left), device=device)

        target_embeddings = self.entity_embedder(batch['entity_ids'])
        replacement_embeddings = self.entity_embedder(entity_replacements)
        target_relation_inputs = torch.cat((target_embeddings[relation_entity_indices_left[selected_relation_indices]], target_embeddings[relation_entity_indices_right[selected_relation_indices]]), dim=-1)

        target_relation_representation = target_relation_inputs

        replacement_embeddings_repeated = replacement_embeddings[torch.arange(batch_size, device=device).repeat_interleave(n_annotations)]

        target_embeddings_expanded = target_embeddings.unsqueeze(1).expand(-1, n_negatives, -1)

        replacement_relation_input_left = torch.cat((replacement_embeddings_repeated,target_embeddings_expanded), dim=-1)

        replacement_relation_input_right = torch.cat((target_embeddings_expanded, replacement_embeddings_repeated), dim=-1)

        replacement_relation_input_self = torch.cat((replacement_embeddings, replacement_embeddings), dim=-1)


        replacement_self_indices = batch['replacement_self_indices']
        replacement_relation_input_left[replacement_self_indices] = replacement_relation_input_self
        replacement_relation_input_right[replacement_self_indices] = replacement_relation_input_self

        replacement_relation_representation_left = replacement_relation_input_left
        replacement_relation_representation_right = replacement_relation_input_right

        replacement_relation_indices_left = batch['replacement_relation_indices_left']
        replacement_relation_indices_right = batch['replacement_relation_indices_right']



        target_product = target_relation_representation * query_relation_representation
        target_sum = target_product.sum(-1)

        # sum element-wise product
        target_relation_compatibility_scores = target_sum * query_relation_scores
        put_indices = torch.arange(batch_size, device=device).repeat_interleave(n_annotations * n_annotations)[selected_relation_indices]

        # sum relation scores within sample
        target_scores = torch.zeros(batch_size, device=device, dtype=float_type)
        target_scores = target_scores.index_put((put_indices,), target_relation_compatibility_scores[selected_relation_indices], accumulate=True)

        replacement_product_left = replacement_relation_representation_left * query_relation_representation[replacement_relation_indices_left].unsqueeze(1)
        replacement_sum_left = replacement_product_left.sum(-1)

        replacement_product_right = replacement_relation_representation_right * query_relation_representation[replacement_relation_indices_right].unsqueeze(1)
        replacement_sum_right = replacement_product_right.sum(-1)

        negative_sum = target_sum.clone().unsqueeze(1).expand(-1, n_negatives)

        negative_sum[replacement_relation_indices_left] = replacement_sum_left
        negative_sum[replacement_relation_indices_right] = replacement_sum_right

        negative_relation_compatibility_scores = negative_sum * query_relation_scores.unsqueeze(1)

        # sum relation scores within sample
        negative_scores = torch.zeros(size=(batch_size, n_negatives), device=device, dtype=float_type)
        negative_scores = negative_scores.index_put((put_indices,), negative_relation_compatibility_scores[selected_relation_indices], accumulate=True)

        scores = torch.cat((target_scores.unsqueeze(1), negative_scores), dim=-1)

        return scores

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        parser.add_argument('--triplet_type', type=str, default=None,
                            help='type of triplet model to use for inference')

        RobertaWrapper.add_args(parser)

    @classmethod
    def build_model(cls, args, task, encoder=None):
        if encoder is None:
            encoder = RobertaWrapper.build_model(args, task)
        n_entities = len(task.entity_dictionary)
        return cls(args, encoder, n_entities)


@register_model_architecture('encoder_etp_relation', 'encoder_etp_relation__roberta_base')
def triplet_base_architecture(args):
    base_architecture(args)


@register_model_architecture('encoder_etp_relation', 'encoder_etp_relation__roberta_large')
def triplet_large_architecture(args):
    large_architecture(args)


@register_model_architecture('encoder_etp_relation', 'encoder_etp_relation__roberta_small')
def triplet_small_architecture(args):
    small_architecture(args)
