import torch
from torch.nn.utils.rnn import pad_sequence
import logging

import os
import copy
import numpy as np
import numpy.random as rd
from itertools import chain, permutations

from fairseq.data import data_utils, FairseqDataset
from utils.diagnostic_utils import Diagnostic

from datasets import GraphDataset

logger = logging.getLogger(__name__)


class PMTBDataset(FairseqDataset):

    def __init__(
        self,
        split,
        annotated_text_A,
        annotated_text_B,
        graph_A,
        graph_B,
        similar_entities,
        similarity_scores,
        seed,
        dictionary,
        entity_dictionary,
        k_weak_negs,
        n_tries_entity,
        strong_negatives,
        strong_negative_type,
        negative_temperature,
        replace_tail,
        mutual_positives,
        similar_positives,
        positive_temperature,
    ):
        self.split = split
        self.annotated_text_A = annotated_text_A
        self.annotated_text_B = annotated_text_B
        self.graph_A = graph_A
        self.graph_B = graph_B
        self.similar_entities = similar_entities
        self.similarity_scores = similarity_scores
        self.seed = seed
        self.dictionary = dictionary
        self.entity_dictionary = entity_dictionary

        self.k_weak_negs = k_weak_negs
        self.n_tries_entity = n_tries_entity
        self.replace_tail = replace_tail
        self.strong_negatives = strong_negatives
        self.strong_negative_type = strong_negative_type
        self.negative_temperature = negative_temperature

        self.mutual_positives = mutual_positives
        self.similar_positives = similar_positives
        self.positive_temperature = positive_temperature


        self.epoch = None


    def set_epoch(self, epoch):
        self.graph_A.set_epoch(epoch)
        self.graph_B.set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.graph_A)

    def num_tokens(self, index):
        return self.graph_A.sizes[index]

    def size(self, index):
        return self.graph_A.sizes[index]

    @property
    def sizes(self):
        return self.graph_A.sizes

    def ordered_indices(self):
        return self.graph_A.ordered_indices()

    def get_edge_entities(self, annotation_data, start_block, end_block):
        # From http://sociograph.blogspot.com/2011/12/gotcha-with-numpys-searchsorted.html
        start_block = annotation_data.dtype.type(start_block)
        end_block = annotation_data.dtype.type(end_block)

        # We are interested in all annotations that INTERSECT [start_block; end_block)
        # Recall that the [start_pos; end_pos) interval for the annotation s is defined as
        # [annotations[s - 1][0], annotations[s - 1][1])
        #
        # From https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html
        # side	returned index i satisfies
        # left	a[i-1] < v <= a[i]
        # right	a[i-1] <= v < a[i]
        #
        # First, we need to find an index s such that
        # annotations[s - 1].end_pos <= start_block < annotations[s].end_pos
        s = np.searchsorted(annotation_data[:, 1], start_block, side='right')

        # Second, we need to find an index e such that
        # annotations[e - 1].start_pos < end_block <= annotations[e].start_pos
        e = np.searchsorted(annotation_data[:, 0], end_block, side='left')

        return set(annotation_data[slice(s, e)][:, -1])

    def sample_text(self, headB_tailB_edges, textA, example_class, filter_entities=None):

        # Iterate through edges between headB and tailB (i.e., textB candidates)
        for edge in headB_tailB_edges:

            # For share_one, discard the current edge if it contains entity replace
            if example_class in ['share_one', 'share_none']:
                edge_entities = self.get_edge_entities(
                    self.annotated_text_B.annotation_data.array,
                    edge[GraphDataset.START_BLOCK],
                    edge[GraphDataset.END_BLOCK]
                )
                if np.any(np.array([x in edge_entities for x in filter_entities])):
                    continue

            # Get textB, using the given edge, headB, and tailB
            textB = self.annotated_text_B.annotate_relation(*(edge))

            # Check that textA and textB are not the same (this may occur for positive pairs).
            # If not, return textB.
            if not torch.equal(textA, textB):
                return textB

        return None

    def sample_positive(self, headA, tailA, textA):

        # If replace_tail=True, then always replace tail. Else, randomly choose replace_entity and keep_entity.
        replace_entity = 1 if self.replace_tail else np.random.randint(2)
        keep_entity = 1 - replace_entity
        entity_ids = (headA, tailA)

        # Get edges with keep_entity as the head
        keep_entity_edges = self.graph_B.edges[entity_ids[keep_entity]].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

        if self.mutual_positives:
            # Get replace_entity's neighbors
            replace_entity_neighbors = self.graph_B.edges[entity_ids[replace_entity]].numpy().reshape(-1, GraphDataset.EDGE_SIZE)[:, GraphDataset.TAIL_ENTITY]

            # Get replace_entity's neighbors, excluding keep_entity
            replace_entity_neighbors = np.unique(replace_entity_neighbors[replace_entity_neighbors != entity_ids[keep_entity]])

            # Get all indices of keep_entity's neighbors, for which the neighbor is also one of replace_entity's neighbors
            candidate_edge_idxs = np.flatnonzero(np.in1d(keep_entity_edges[:, GraphDataset.TAIL_ENTITY], replace_entity_neighbors))
        else:
            # Get all indices of keep_entity's neighbors, for which the neighbor is not replace_entity
            candidate_edge_idxs = np.flatnonzero(keep_entity_edges[:, GraphDataset.TAIL_ENTITY] != entity_ids[replace_entity])

        # Check that keep_entity has at least one neighbor besides replace_entity
        if len(candidate_edge_idxs) < 1:
            return None, None, None

        # Get all of entity_keep's edges, excluding those shared with entity_replace
        candidate_edges = keep_entity_edges[candidate_edge_idxs, :]

        # Get entity_replace candidates -- i.e., all of entity_keep's neighbors, excluding entity_replace
        entity_replace_candidates = candidate_edges[:, GraphDataset.TAIL_ENTITY]

        # Get unique array of entity_replace candidates -- i.e., all of entity_keep's neighbors, excluding entity_replace and graph duplicates
        entity_replace_candidates_unique = np.unique(entity_replace_candidates)

        # Set maximum number of replace candidates to consider
        n_entity_replace_candidates = min(self.n_tries_entity, len(entity_replace_candidates_unique))

        # Sample a random array of n_entity_replace_candidates entity_replace candidates
        entity_replace_candidates_sample = entity_replace_candidates_unique[torch.randperm(len(entity_replace_candidates_unique)).numpy()]

        if self.similar_positives:
            replace_entity_id = entity_ids[replace_entity]
            replace_similar_entities = self.similar_entities.array[replace_entity_id]
            replace_similar_entities = replace_similar_entities[replace_similar_entities != -1]
            _, indices_similar, indices_candidates = np.intersect1d(replace_similar_entities, entity_replace_candidates_unique, assume_unique=True,return_indices=True)
            indices_similar = np.sort(indices_similar)
            overlap_entities = replace_similar_entities[indices_similar]
            if len(overlap_entities) > 0:
                overlap_scores = self.similarity_scores.array[replace_entity_id][indices_similar]
                exp_scores = np.exp(overlap_scores/self.positive_temperature)
                overlap_probabilities = exp_scores / exp_scores.sum()
                entity_order = np.random.choice(len(overlap_probabilities), size=len(overlap_probabilities), replace=False, p=overlap_probabilities)
                entity_replace_candidates_sample = np.concatenate((overlap_entities[entity_order], entity_replace_candidates_sample))

        entity_replace_candidates_sample = entity_replace_candidates_sample[:n_entity_replace_candidates]

        # Iterate through all of the entity_replace candidates
        for entity_replace_candidate in entity_replace_candidates_sample:
            candidate_head = headA if keep_entity == 0 else entity_replace_candidate
            candidate_tail = tailA if keep_entity == 1 else entity_replace_candidate

            candidate_edges = self.graph_B.edges[candidate_head].numpy().reshape(-1, GraphDataset.EDGE_SIZE)
            # Get all edges between keep_entity and entity_replace_candidate, according to the shuffled indices
            replace_edges = candidate_edges[candidate_edges[:, GraphDataset.TAIL_ENTITY] == candidate_tail]

            # Shuffle replace_edges
            replace_edges = replace_edges[torch.randperm(len(replace_edges)).numpy()]

            # Sample textB from replace_edges
            textB = self.sample_text(replace_edges, textA, 'share_one', [entity_ids[replace_entity]])
            if textB is not None:
                break

        return textB, entity_ids[keep_entity], entity_replace_candidate

    def sample_strong_negative(self, headA, tailA, textA, pos_keep_entity, pos_new_entity, strong_negative_type):

        # Get edges with headA/tailA/pos_new_entity as the head
        headA_edges = self.graph_B.edges[headA].numpy().reshape(-1, GraphDataset.EDGE_SIZE)
        tailA_edges = self.graph_B.edges[tailA].numpy().reshape(-1, GraphDataset.EDGE_SIZE)
        pos_new_entity_edges = self.graph_B.edges[pos_new_entity].numpy().reshape(-1, GraphDataset.EDGE_SIZE)
        if strong_negative_type == 'similarity':
            new_entity_neighbors = self.graph_B.get_neighbors(pos_new_entity)
            new_entity_similar = self.similar_entities.array[pos_new_entity]
            new_entity_similar = new_entity_similar[new_entity_similar != -1]
            _, neighbor_indices_overlap, similar_indices_overlap = np.intersect1d(new_entity_neighbors, new_entity_similar, assume_unique=True, return_indices=True)
            similar_indices_overlap = np.sort(similar_indices_overlap)

            overlap_entities = new_entity_similar[similar_indices_overlap]

            if len(overlap_entities) == 0:
                return None, None

            overlap_scores = self.similarity_scores.array[pos_new_entity][similar_indices_overlap]
            exp_scores = np.exp(overlap_scores/self.negative_temperature)
            overlap_probabilities = exp_scores / exp_scores.sum()
            entity_order = np.random.choice(len(overlap_probabilities), size=len(overlap_probabilities), replace=False, p=overlap_probabilities)
            negative_entity_candidates = overlap_entities[entity_order]

        # Given AB (base) and AC/CB (positive), where C is a mutual neighbor of A and B
        # Sample DC/CD (strong negative), where D is also a mutual neighbor of A and B
        if strong_negative_type == 'mutual':
            # Get tail neighbors of headA/tailA
            headA_neighbors = np.unique(headA_edges[:, GraphDataset.TAIL_ENTITY])
            tailA_neighbors = np.unique(tailA_edges[:, GraphDataset.TAIL_ENTITY])

            # Get pos_new_entity edge indices for which the tail is a mutual neighbor of headA and tailA
            pos_new_entity_headA_candidate_edge_indxs = np.flatnonzero(np.in1d(pos_new_entity_edges[:, GraphDataset.TAIL_ENTITY], headA_neighbors))
            pos_new_entity_tailA_candidate_edge_indxs = np.flatnonzero(np.in1d(pos_new_entity_edges[:, GraphDataset.TAIL_ENTITY], tailA_neighbors))
            pos_new_entity_candidate_edge_indxs = np.intersect1d(pos_new_entity_headA_candidate_edge_indxs, pos_new_entity_tailA_candidate_edge_indxs)

            # Check that there is at least one entity that is a mutual neighbor of pos_new_entity, headA, and tailA
            if len(pos_new_entity_candidate_edge_indxs) < 1:
                return None, None

            # Get all possible strong negative entity candidates
            negative_entity_candidates = np.unique(pos_new_entity_edges[pos_new_entity_candidate_edge_indxs][:, GraphDataset.TAIL_ENTITY])

            # Sample a random array of n_newB_sneg_candidates entity candidates
            negative_entity_candidates = negative_entity_candidates[torch.randperm(len(negative_entity_candidates)).numpy()]

        # Set maximum number of entity candidates to consider
        n_entity_negative_candidates = min(self.n_tries_entity, len(negative_entity_candidates))
        negative_entity_candidates = negative_entity_candidates[:n_entity_negative_candidates]

        # Iterate through the strong negative entity pair candidates
        textB_strong_neg = None
        for negative_entity in negative_entity_candidates:
            headB_strong_neg = pos_new_entity if pos_keep_entity == headA else negative_entity
            tailB_strong_neg = pos_new_entity if pos_keep_entity == tailA else negative_entity

            # Get edges with headB_strong_neg as the head
            headB_strong_neg_edges = self.graph_B.edges[headB_strong_neg].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

            # Get edges with (headB_strong_neg, tailB_strong_neg) as (head, tail)
            headB_tailB_strong_neg_edges = headB_strong_neg_edges[headB_strong_neg_edges[:, GraphDataset.TAIL_ENTITY] == tailB_strong_neg]

            # Check that there is at least one edge with (headB_strong_neg, tailB_strong_neg) as (head, tail)
            if len(headB_tailB_strong_neg_edges) < 1:
                continue

            # Shuffle headB_tailB_strong_neg_edges
            headB_tailB_strong_neg_edges = headB_tailB_strong_neg_edges[torch.randperm(len(headB_tailB_strong_neg_edges)).numpy()]

            # Sample textB_strong_neg from headB_tailB_strong_neg_edges
            textB_strong_neg = self.sample_text(headB_tailB_strong_neg_edges, textA, 'share_none', [headA, tailA])
            if textB_strong_neg is not None:
                break

        return textB_strong_neg, negative_entity

    def __getitem__(self, index):
        edge = self.graph_A[index]
        headA = edge[GraphDataset.HEAD_ENTITY].item()
        tailA = edge[GraphDataset.TAIL_ENTITY].item()

        with data_utils.numpy_seed(9031935, self.seed, self.epoch, index):
            textA = self.annotated_text_A.annotate_relation(*(edge.numpy()))

            # Sample positive text pair: textA and textB share one target entity
            textB_pos, pos_keep_entity, pos_replace_entity = self.sample_positive(headA, tailA, textA)

            # Check if positive text pair was successfully sampled
            if textB_pos is None:
                return None

            # Initialize textB list with textB_pos
            textB = [textB_pos]

            if self.strong_negatives:
                # Sample one strong negative text pair
                textB_strong_neg, negative_entity = self.sample_strong_negative(headA, tailA, textA, pos_keep_entity, pos_replace_entity, self.strong_negative_type)

                # Check if strong negative text pair was successfully sampled
                if textB_strong_neg is None:
                    return None

                # Append textB_strong_neg to textB list
                textB.append(textB_strong_neg)

        item = {
            'textA': textA,
            'textB': textB,
            'headA': headA,
            'tailA': tailA,
            'pos_keep_entity': pos_keep_entity,
            'pos_replace_entity': pos_replace_entity,
            'ntokens': len(textA),
            'nsentences': 1,
            'ntokens_AB': len(textA) + sum([len(x) for x in textB]),
        }

        if self.strong_negatives:
            item['negative_entity'] = negative_entity

        return item

    def collater(self, instances):
        # Filter out instances for which no positive text pair exists
        instances = [x for x in instances if x is not None]

        # Get batch size
        batch_size = len(instances)
        if batch_size == 0:
            return None

        # Get initial number of textBs per instance
        n_textB_init = 2 if self.strong_negatives else 1

        # Initialize lists, dicts, and counters
        textA_list, textB_dict, A2B_dict = [], {}, {}
        ntokens, nsentences, ntokens_AB = 0, 0, 0

        # Get headA and tailA lists
        headA_list = np.array([instance['headA'] for instance in instances])
        tailA_list = np.array([instance['tailA'] for instance in instances])

        # Get pos_keep_entity and pos_replace_entity lists
        if n_textB_init == 1:
            fixB_list = np.array([instance['pos_keep_entity'] for instance in instances])
            newB_list = np.array([instance['pos_replace_entity'] for instance in instances])
        else:
            fixB_list = np.array(list(chain.from_iterable([[instance['pos_keep_entity'], instance['pos_replace_entity']] for instance in instances])))
            newB_list = np.array(list(chain.from_iterable([[instance['pos_replace_entity'], instance['negative_entity']] for instance in instances])))

        # Get array of textB lengths
        textB_len = np.array(list(chain.from_iterable([[len(t) for t in instance['textB']] for instance in instances])))

        # Compute statistics for textB lengths
        textB_mean = np.mean(textB_len)
        textB_std = np.std(textB_len)

        # Generate cluster candidates based on textB lengths statistics
        bin_vals = [0] + [textB_mean + 0.5*k*textB_std for k in range(-3, 4)] + [float('inf')]
        cluster_candidates = [np.where(np.logical_and(textB_len > bin_vals[i], textB_len <= bin_vals[i+1]))[0] for i in range(len(bin_vals)-1)]

        # Build textB clusters; initialize textB_dict and A2B_dict
        cluster_id = 0
        textB_clusters = -1 * np.ones(batch_size * n_textB_init)
        for c in cluster_candidates:
            if len(c) > 0:
                textB_clusters[c] = cluster_id
                textB_dict[cluster_id] = []
                A2B_dict[cluster_id] = []
                cluster_id += 1

        # Populate textA_list, textB_dict, and other auxiliary lists
        for i, instance in enumerate(instances):
            textA_list.append(instance['textA'])
            for j, cur_textB in enumerate(instance['textB']):
                cluster_id = textB_clusters[i * n_textB_init + j]
                textB_dict[cluster_id].append(cur_textB)
                A2B_dict[cluster_id].append(i * n_textB_init + j)

            ntokens += instance['ntokens']
            nsentences += instance['nsentences']
            ntokens_AB += instance['ntokens_AB']

        # Pad textA
        padded_textA = pad_sequence(textA_list, batch_first=True, padding_value=self.dictionary.pad())

        # Pad textB, and get A2B_list
        padded_textB = {}
        padded_textB_size = 0
        A2B_list = []
        for cluster_id, cluster_texts in textB_dict.items():
            padded_textB[cluster_id] = pad_sequence(cluster_texts, batch_first=True, padding_value=self.dictionary.pad())
            padded_textB_size += torch.numel(padded_textB[cluster_id])
            A2B_list += A2B_dict[cluster_id]
        A2B_list = np.argsort(A2B_list)
        A2B = A2B_list.reshape(batch_size, n_textB_init)

        # Add k weak negatives (i.e., negatives not guaranteed to be strong)
        # to each positive, using texts in the current batch
        k_weak_negs = min(self.k_weak_negs, batch_size * n_textB_init - n_textB_init)
        A2B_weak_negs = -1 * np.ones((batch_size, k_weak_negs))
        textB_idxs = np.arange(batch_size * n_textB_init)
        bad_weak_negs = 0
        for i in range(batch_size):
            self_textB_cond = np.logical_and(textB_idxs != i*n_textB_init, textB_idxs != i*n_textB_init+1) if self.strong_negatives else textB_idxs != i
            weak_neg_conditions = np.logical_and.reduce(
                (
                    self_textB_cond,
                    np.logical_not(np.logical_or(fixB_list == headA_list[i], newB_list == tailA_list[i])),
                    np.logical_not(np.logical_or(fixB_list == tailA_list[i], newB_list == headA_list[i]))
                )
            )
            weak_neg_candidates = A2B_list[np.flatnonzero(weak_neg_conditions)]
            cur_bad_weak_negs = batch_size * n_textB_init - n_textB_init - len(weak_neg_candidates)
            bad_weak_negs += cur_bad_weak_negs
            weak_negs_init = weak_neg_candidates[torch.randperm(len(weak_neg_candidates)).numpy()]

            weak_negs = weak_negs_init
            while len(weak_negs) < k_weak_negs:
                weak_negs = np.concatenate((weak_negs, weak_negs_init)) # pad to make up for discarded weak negs
            weak_negs = weak_negs[:k_weak_negs]
            A2B_weak_negs[i, :] = weak_negs
        A2B = np.concatenate((A2B, A2B_weak_negs), axis=1).flatten()

        batch_dict = {
            'textA': padded_textA,
            'textB': padded_textB,
            'A2B': torch.LongTensor(A2B),
            'target': torch.zeros(batch_size, dtype=torch.long),
            'size': batch_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'ntokens_AB': ntokens_AB,
            'ntokens_mem': padded_textA.numel() + padded_textB_size,
            'bad_weak_negs': bad_weak_negs / batch_size
        }

        return batch_dict