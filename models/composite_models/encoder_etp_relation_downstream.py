import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import models

from fairseq.models import register_model, register_model_architecture
from fairseq.models import BaseFairseqModel

import tasks
from models.encoder.roberta import RobertaWrapper, base_architecture, large_architecture, small_architecture
from utils.misc_utils import unravel_index


@register_model('encoder_etp_relation_downstream')
class EncoderETPRelationDownstream(BaseFairseqModel):

    def __init__(self, args, encoder, n_entities):
        super().__init__()

        self.args = args

        self.entity_dim = args.entity_dim
        self.encoder_embed_dim = args.encoder_embed_dim

        self.encoder = encoder
        self.entity_embedder = nn.Embedding(n_entities, args.entity_dim)
        self.choice_size = args.choice_size
        self.beam_size = args.beam_size

    def forward(self, batch):

        n_annotations_inclusive = batch['n_annotations']
        n_annotations = n_annotations_inclusive - 1
        batch_size = len(n_annotations)
        device = n_annotations.device
        entity_ids = batch['entity_ids']
        target = batch['target']

        relation_entity_indices_left = batch['relation_entity_indices_left']
        relation_entity_indices_right = batch['relation_entity_indices_right']

        # attention - all_annotations include mask?
        encoder_output, _ = self.encoder(batch['text'], mask_annotation=batch['mask_annotation'], all_annotations=batch['all_annotations'], n_annotations=n_annotations_inclusive, relation_entity_indices_left=relation_entity_indices_left, relation_entity_indices_right=relation_entity_indices_right)
        query_entity_representation_inclusive, query_relation_representation_inclusive, query_relation_scores_inclusive = encoder_output

        float_type = query_entity_representation_inclusive.dtype
        int_type = relation_entity_indices_left.dtype

        mask_entity_indices = n_annotations.cumsum(0) - 1
        non_mask_bool_entity = torch.ones(len(query_entity_representation_inclusive), dtype=int_type, device=device)
        non_mask_bool_entity[mask_entity_indices] = 0
        non_mask_entity_indices = non_mask_bool_entity.nonzero(as_tuple=True)[0]

        mask_relation_indices = batch['mask_relation_indices']
        mask_throw_away_indices = batch['mask_throw_away_indices']
        non_mask_bool_relation = torch.ones(len(query_relation_representation_inclusive), dtype=int_type, device=device)
        non_mask_bool_relation[mask_relation_indices] = 0
        non_mask_bool_relation[mask_throw_away_indices] = 0
        non_mask_relation_indices = non_mask_bool_relation.nonzero(as_tuple=True)[0]

        mask_entity_representation = query_entity_representation_inclusive[mask_entity_indices]
        mask_relation_representation = query_relation_representation_inclusive[mask_relation_indices]
        mask_relation_scores = query_relation_scores_inclusive[mask_relation_indices]

        query_entity_representation = query_entity_representation_inclusive[non_mask_entity_indices]
        query_relation_representation = query_relation_representation_inclusive[non_mask_relation_indices]
        query_relation_scores = query_relation_scores_inclusive[non_mask_relation_indices]

        entity_ends = n_annotations.cumsum(0)

        entity_starts = torch.zeros_like(entity_ends)
        entity_starts[1:] = n_annotations[:-1].cumsum(0)
        relation_offsets = torch.zeros_like(entity_ends)
        relation_offsets[1:] = (n_annotations * n_annotations).cumsum(0)[:-1]

        mask_relation_offsets = (n_annotations * 2).cumsum(0) - 2 * n_annotations

        # attention - query entity representation layer?
        entity_scores = torch.einsum('ij,kj->ik', [query_entity_representation, self.entity_embedder.weight])
        top_entity_scores, top_entities = entity_scores.topk(self.choice_size, dim=1, sorted=True)

        mask_entity_scores = torch.einsum('ij,kj->ik', [mask_entity_representation, self.entity_embedder.weight])
        top_mask_entity_scores, top_mask_entities = mask_entity_scores.topk(self.choice_size, dim=1, sorted=True)

        # # Add correct answer to top mask entities.

        prediction_equals_target = (top_mask_entity_scores - target.unsqueeze(1))
        contained_sample_idx, contained_target_choice_idx = torch.nonzero((prediction_equals_target==0), as_tuple=True)
        not_contained_sample_idx, _ = torch.nonzero((prediction_equals_target), as_tuple=True)
        not_contained_sample_idx = torch.unique_consecutive(not_contained_sample_idx)

        put_indices = (not_contained_sample_idx, torch.empty_like(not_contained_sample_idx).fill_(self.choice_size-1))

        top_mask_entities = top_mask_entities.index_put(put_indices, target[not_contained_sample_idx])

        # If in train mode, add with its score, otherwise add with negative inf to avoid cheating validation
        if self.training:
            top_mask_entity_scores = top_mask_entity_scores.index_put(put_indices, mask_entity_scores[(not_contained_sample_idx, target[not_contained_sample_idx])])
        else:
            top_mask_entity_scores = top_mask_entity_scores.index_put(put_indices, torch.empty(len(not_contained_sample_idx), dtype=float_type, device=device).fill_(-1e8))


        target_choice_idx = torch.zeros(batch_size, dtype=int_type, device=device)
        target_choice_idx[contained_sample_idx] = contained_target_choice_idx
        target_choice_idx[not_contained_sample_idx] = self.choice_size-1

        # WARNING SIDE EFFECTS BAD MICHIEL, BAD
        batch['target'] = target_choice_idx

        target_scores = torch.zeros((batch_size, self.beam_size), dtype=float_type, device=device)

        for idx in range(batch_size):
            entity_start = entity_starts[idx]
            entity_end = entity_ends[idx]
            n_ent = entity_end - entity_start

            relation_idx = torch.arange(n_ent**2, dtype=int_type, device=device).reshape(n_ent, n_ent) + relation_offsets[idx]

            threshold = 0.5 * (query_relation_scores[relation_idx].max() + query_relation_scores[relation_idx].mean())
            indices_selected_boolean = query_relation_scores[relation_idx] > threshold

            relation_score_normalization = (2 * self.args.entity_dim * n_ent**2)
            entity_score_normalization = (self.args.entity_dim * n_ent)

            # first step calculate scores once for beam=1 to avoid duplicates

            beam = entity_ids[entity_start:entity_end].clone()
            choice_scores = top_entity_scores[entity_start:entity_end]
            n_assigned = len(torch.nonzero((beam >= 0)))

            # Just do entity prediction if no other annotations exist
            if n_annotations[idx] == 0:
                choice_scores = (top_mask_entity_scores[idx] / entity_score_normalization).unsqueeze(0)
                target_scores = target_scores.index_copy(0, torch.empty(1, dtype=int_type, device=device).fill_(idx), choice_scores)
                continue


            if n_assigned > 0:

                assigned_entities = torch.nonzero((beam>=0)).reshape(n_assigned, 1).expand( -1, n_ent).reshape(-1)
                choice_positions = torch.arange(n_ent, dtype=int_type, device=device).unsqueeze(0).expand(n_assigned, -1).reshape(-1)

                relation_indices_local = (torch.cat((assigned_entities, choice_positions)), torch.cat((choice_positions, assigned_entities)))
                indices_selected_local = indices_selected_boolean[relation_indices_local].nonzero(as_tuple=True)[0]

                indices_selected_global = relation_idx[relation_indices_local][indices_selected_local]

                initial_query_relation_representation = query_relation_representation[indices_selected_global]
                initial_query_relation_scores = query_relation_scores[indices_selected_global]

                current_embedding_ids = beam[(beam>=0).nonzero(as_tuple=True)].reshape(n_assigned, 1, 1).expand(-1, n_ent, self.choice_size).reshape(-1, self.choice_size)
                choice_embedding_ids = top_entities[entity_start:entity_end].reshape(1, n_ent, self.choice_size).expand(n_assigned, -1, -1).reshape(-1, self.choice_size)
                initial_kb_relation_indices_left = torch.stack((current_embedding_ids, choice_embedding_ids), dim=-1)
                initial_kb_relation_indices_right = torch.stack((choice_embedding_ids, current_embedding_ids), dim=-1)
                initial_kb_relation_indices = torch.cat((initial_kb_relation_indices_left, initial_kb_relation_indices_right))

                initial_kb_relation_indices_selected = initial_kb_relation_indices[indices_selected_local]

                initial_kb_relation_representation = self.entity_embedder(initial_kb_relation_indices_selected).reshape(-1, self.choice_size, 2 * self.entity_dim)

                initial_relation_compatibility_scores = torch.einsum('rd,rcd->rc', [initial_query_relation_representation * initial_query_relation_scores.unsqueeze(1), initial_kb_relation_representation]) / relation_score_normalization

                initial_relation_put_indices = (torch.cat((choice_positions, choice_positions))[indices_selected_local],)
                choice_scores = choice_scores.index_put(initial_relation_put_indices, initial_relation_compatibility_scores, accumulate=True)



                # .reshape(n_assigned, 1, 1, 1, 1).expand(-1, n_ent, -1, self.choice_size, -1)
                # choice_embedding_ids = top_entities[entity_start:entity_end].reshape(1, n_ent, 1, self.choice_size, 1).expand(n_assigned, -1, -1, -1, -1)

                # initial_kb_relation_indices_left = torch.cat((current_embedding_ids, choice_embedding_ids), dim=-1)
                # initial_kb_relation_indices_right = torch.cat((choice_embedding_ids, current_embedding_ids), dim=-1)


                # initial_relation_indices_left = relation_idx[assigned_entities, choice_positions].reshape(n_assigned, n_ent, 1)
                # initial_relation_indices_right = relation_idx[choice_positions, assigned_entities].reshape(n_assigned, n_ent, 1)

                # initial_relation_indices = torch.cat((initial_relation_indices_left, initial_relation_indices_right), dim=-1)

                # initial_query_relation_representation = query_relation_representation[initial_relation_indices]
                # initial_query_relation_scores = query_relation_scores[initial_relation_indices]


                # current_embedding_ids = beam[(beam>=0).nonzero(as_tuple=True)].reshape(n_assigned, 1, 1, 1, 1).expand(-1, n_ent, -1, self.choice_size, -1)
                # choice_embedding_ids = top_entities[entity_start:entity_end].reshape(1, n_ent, 1, self.choice_size, 1).expand(n_assigned, -1, -1, -1, -1)

                # initial_kb_relation_indices_left = torch.cat((current_embedding_ids, choice_embedding_ids), dim=-1)
                # initial_kb_relation_indices_right = torch.cat((choice_embedding_ids, current_embedding_ids), dim=-1)

                # initial_kb_relation_indices = torch.cat((initial_kb_relation_indices_left, initial_kb_relation_indices_right), dim=-3)

                # # only selected indices

                # initial_kb_relation_representation = self.entity_embedder(initial_kb_relation_indices).reshape(n_assigned, n_ent, 2, self.choice_size, -1)

                # initial_relation_scores = torch.einsum('jkld,jklcd->jklc', [initial_query_relation_representation, initial_kb_relation_representation])

                # choice_scores = choice_scores + torch.einsum('jkl,jklc->kc', [initial_query_relation_scores, initial_relation_scores]) / relation_score_normalization

                non_zero_put_indices = (beam>=0).nonzero(as_tuple=True)
                # mask scores of already assigned positions
                choice_scores = choice_scores.index_copy(0, non_zero_put_indices[0], torch.empty((len(non_zero_put_indices[0]), self.choice_size), dtype=float_type, device=device).fill_(float('-inf')))

            score_shape = choice_scores.shape
            # reshape
            flat_choice_scores = choice_scores.reshape(-1)

            # topk
            best_scores, flat_best_score_indices = flat_choice_scores.topk(self.beam_size, sorted=False)

            # create new beam
            best_score_indices = unravel_index(flat_best_score_indices, score_shape)

            # convert choice indices to entity indices
            best_position_indices = best_score_indices[:, 0]
            best_entity_ids = top_entities[(best_position_indices + entity_start, best_score_indices[:, 1])]

            # populate beam
            beam = beam.unsqueeze(0).expand(self.beam_size, -1).contiguous()
            beam = beam.index_put((torch.arange(self.beam_size, dtype=int_type, device=device), best_position_indices), best_entity_ids)

            beam_scores = best_scores

            start_step = len(torch.nonzero((beam[0] >= 0)))
            for step in range(start_step, n_ent):

                non_zero_indices = torch.nonzero((beam>=0), as_tuple=True)
                assigned_entities = non_zero_indices[1].reshape(-1, step, 1).expand(-1, -1, n_ent).reshape(-1)

                choice_positions = torch.arange(n_ent, dtype=int_type, device=device).unsqueeze(0).unsqueeze(0).expand(self.beam_size, step, -1).reshape(-1)

                relation_indices_local = (torch.cat((assigned_entities, choice_positions)), torch.cat((choice_positions, assigned_entities)))
                indices_selected_local = indices_selected_boolean[relation_indices_local].nonzero(as_tuple=True)[0]

                indices_selected_global = relation_idx[relation_indices_local][indices_selected_local]

                step_query_relation_representation = query_relation_representation[indices_selected_global]
                step_query_relation_scores = query_relation_scores[indices_selected_global]

                current_embedding_ids = beam[(beam>=0).nonzero(as_tuple=True)].reshape(self.beam_size, step, 1, 1).expand(-1, -1, n_ent, self.choice_size).reshape(-1, self.choice_size)
                choice_embedding_ids = top_entities[entity_start:entity_end].reshape(1, 1, n_ent, self.choice_size).expand(self.beam_size, step, -1, -1).reshape(-1, self.choice_size)

                step_kb_relation_indices_left = torch.stack((current_embedding_ids, choice_embedding_ids), dim=-1)
                step_kb_relation_indices_right = torch.stack((choice_embedding_ids, current_embedding_ids), dim=-1)
                step_kb_relation_indices = torch.cat((step_kb_relation_indices_left, step_kb_relation_indices_right))

                step_kb_relation_indices_selected = step_kb_relation_indices[indices_selected_local]

                step_kb_relation_representation = self.entity_embedder(step_kb_relation_indices_selected).reshape(-1, self.choice_size, 2 * self.entity_dim)


                step_relation_compatibility_scores = torch.einsum('rd,rcd->rc', [step_query_relation_representation * step_query_relation_scores.unsqueeze(1), step_kb_relation_representation]) / relation_score_normalization

                beam_indices_half = torch.arange(self.beam_size, device=device).repeat_interleave(n_ent * step)
                beam_indices = torch.cat((beam_indices_half, beam_indices_half))[indices_selected_local]
                position_indices = torch.cat((choice_positions, choice_positions))[indices_selected_local]
                step_relation_put_indices = (beam_indices, position_indices)

                choice_scores = beam_scores.reshape(self.beam_size, 1, 1).expand(-1, n_ent, self.choice_size).contiguous()
                choice_scores = choice_scores.index_put(step_relation_put_indices, step_relation_compatibility_scores, accumulate=True)

                # step_relation_indices_left = relation_idx[assigned_entities, choice_positions].reshape(self.beam_size, step, n_ent, 1)
                # step_relation_indices_right = relation_idx[choice_positions, assigned_entities].reshape(self.beam_size, step, n_ent, 1)

                # step_relation_indices = torch.cat((step_relation_indices_left, step_relation_indices_right), dim=-1)

                # step_query_relation_representation = query_relation_representation[step_relation_indices]

                # step_query_relation_scores = query_relation_scores[step_relation_indices]

                # current_embedding_ids = beam[(beam>=0).nonzero(as_tuple=True)].reshape(self.beam_size, step, 1, 1, 1).expand(-1, -1, n_ent, -1, self.choice_size)
                # choice_embedding_ids = top_entities[entity_start:entity_end].reshape(1, 1, n_ent, 1, self.choice_size).expand(self.beam_size, step, -1, -1, -1)

                # current_embedding_values = self.entity_embedder(current_embedding_ids)
                # choice_embedding_values = self.entity_embedder(choice_embedding_ids)

                # step_kb_relation_representation_left = torch.cat((current_embedding_values, choice_embedding_values), dim=-1)

                # step_kb_relation_representation_right = torch.cat((current_embedding_values, choice_embedding_values), dim=-1)

                # step_kb_relation_representation = torch.cat((step_kb_relation_representation_left, step_kb_relation_representation_right), dim=-3)

                # step_relation_scores = torch.einsum('ijkld,ijklcd->ijklc', [step_query_relation_representation, step_kb_relation_representation])

                # choice_scores = torch.einsum('ijkl,ijklc->ikc', [step_query_relation_scores, step_relation_scores]) / relation_score_normalization
                choice_scores = choice_scores + top_entity_scores[entity_start:entity_end].unsqueeze(0).expand(self.beam_size, -1, -1) / entity_score_normalization
                # choice_scores = choice_scores + beam_scores


                # mask scores of already assigned positions
                non_zero_put_indices = (beam>=0).nonzero(as_tuple=True)
                # mask scores of already assigned positions
                choice_scores = choice_scores.index_put((beam>=0).nonzero(as_tuple=True), torch.empty((len(non_zero_put_indices[0]), self.choice_size), dtype=float_type, device=device).fill_(float('-inf')))

                score_shape = choice_scores.shape
                # reshape
                flat_choice_scores = choice_scores.reshape(-1)

                # topk
                best_scores, flat_best_score_indices = flat_choice_scores.topk(self.beam_size, sorted=False)

                # create new beam
                best_score_indices = unravel_index(flat_best_score_indices, score_shape)

                # new beam scores are just best scores
                beam_scores = best_scores

                # convert choice indices to entity indices
                best_position_indices = best_score_indices[:, 1]
                best_entity_ids = top_entities[(best_position_indices + entity_start, best_score_indices[:, 2])]

                beam = beam.index_select(0, best_score_indices[:, 0])

                # populate beam
                beam = beam.index_put((torch.arange(self.beam_size, dtype=int_type, device=device), best_position_indices), best_entity_ids)


            # decode most likely mask positions, add answer

            current_embedding_ids = beam.reshape(self.beam_size, n_ent, 1, 1).expand(-1, -1, -1, self.choice_size)
            choice_embedding_ids = top_mask_entities[idx].reshape(1, 1, 1, self.choice_size).expand(self.beam_size, n_ent, -1, -1)

            current_embedding_values = self.entity_embedder(current_embedding_ids)
            choice_embedding_values = self.entity_embedder(choice_embedding_ids)

            kb_relation_representation_left = torch.cat((current_embedding_values, choice_embedding_values), dim=-1)
            kb_relation_representation_right = torch.cat((choice_embedding_values, current_embedding_values), dim=-1)

            kb_relation_representation = torch.cat((kb_relation_representation_left, kb_relation_representation_right), dim=-3)

            instance_mask_relation_indices_left = (torch.arange(n_annotations[idx], device=device) + mask_relation_offsets[idx]).reshape(n_ent, 1)
            instance_mask_relation_indices_right = (torch.arange(n_annotations[idx], device=device) + n_annotations[idx] + mask_relation_offsets[idx]).reshape(n_ent, 1)
            instance_mask_relation_indices = torch.cat((instance_mask_relation_indices_left, instance_mask_relation_indices_right), dim=-1)

            instance_mask_relation_representation = mask_relation_representation[instance_mask_relation_indices]
            instance_mask_relation_scores = mask_relation_scores[instance_mask_relation_indices]

            instance_scores = torch.einsum('bpacd,pad->bpac', [kb_relation_representation, instance_mask_relation_representation])

            instance_scores = torch.einsum('pa,bpac->bc', [instance_mask_relation_scores, instance_scores]) / relation_score_normalization

            choice_scores = beam_scores.unsqueeze(1) + instance_scores
            choice_scores = choice_scores + top_mask_entity_scores[idx] / entity_score_normalization
            choice_scores = choice_scores.max(0, keepdim=True).values

            target_scores = target_scores.index_copy(0, torch.empty(1, dtype=int_type, device=device).fill_(idx), choice_scores)

        return target_scores

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


@register_model_architecture('encoder_etp_relation_downstream', 'encoder_etp_relation_downstream__roberta_base')
def triplet_base_architecture(args):
    base_architecture(args)


@register_model_architecture('encoder_etp_relation_downstream', 'encoder_etp_relation_downstream__roberta_large')
def triplet_large_architecture(args):
    large_architecture(args)


@register_model_architecture('encoder_etp_relation_downstream', 'encoder_etp_relation_downstream__roberta_small')
def triplet_small_architecture(args):
    small_architecture(args)
