import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import logging

import os
import copy
import numpy as np
import numpy.random as rd
import itertools

from fairseq.data import data_utils, FairseqDataset
from utils.diagnostic_utils import Diagnostic

from datasets import GraphDataset

logger = logging.getLogger(__name__)


class BoRDataset(FairseqDataset):

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
        n_strong_candidates,
        n_weak_candidates,
        head_tail_weight,
        n_tries_entity,
    ):
        self.split = split
        self.annotated_text_A = annotated_text_A
        self.annotated_text_B = annotated_text_B
        self.graph_A = graph_A
        self.graph_B = graph_B

        self.similar_entities = similar_entities
        self.similarity_scores = similarity_scores
        n_entities = len(self.similar_entities.array)
        self.similar_entities.array = np.concatenate((
            np.expand_dims(np.arange(n_entities), 1),
            self.similar_entities.array
        ), axis=1)
        self.similarity_scores.array = np.concatenate((
            np.ones((n_entities, 1)),
            self.similarity_scores.array
        ), axis=1)

        self.seed = seed
        self.dictionary = dictionary

        self.n_strong_candidates = n_strong_candidates
        assert self.n_strong_candidates >= 2
        self.n_weak_candidates = n_weak_candidates
        self.head_tail_weight = head_tail_weight
        self.n_tries_entity = n_tries_entity

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

    def sample_text(self, headB_tailB_edges, textA, example_class, filter_entities=[]):

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


    def sample_bor_candidates(self, headA, tailA, textA):

        # Randomly assign entity_replace and entity_keep
        entity_replace = np.random.randint(2)
        entity_keep = 1 - entity_replace
        entity_ids = (headA, tailA)

        # Get edges with entity_keep as the head
        entity_keep_edges = self.graph_B.edges[entity_ids[entity_keep]].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

        # Get entity_replace candidates -- i.e., all of entity_keep's neighbors, including duplicates
        entity_replace_candidates = entity_keep_edges[:, GraphDataset.TAIL_ENTITY]

        # Get unique array of entity_replace candidates -- i.e., all of entity_keep's neighbors, excluding duplicates
        entity_replace_candidates_unique = np.unique(entity_replace_candidates)

        # Keep only entity_replace candidates which are also among entity_replace's most similar entities
        entity_replace_candidates_unique, _, entity_replace_candidates_idx = np.intersect1d(entity_replace_candidates_unique, self.similar_entities.array[entity_replace], assume_unique=False, return_indices=True)

        # Sort entity_replace_candidates_idx and entity_replace_candidates_unique in descending order w.r.t. score
        entity_replace_candidates_idx = np.sort(entity_replace_candidates_idx)
        entity_replace_candidates_unique = self.similar_entities.array[entity_replace][entity_replace_candidates_idx]

        # Check that entity_keep has at least n_strong_candidates neighbors
        if len(entity_replace_candidates_idx) < self.n_strong_candidates:
            return None, None, None

        # Split entity_replace_candidates_idx and entity_replace_candidates_unique into n_strong_candidates chunks
        entity_replace_candidates_chunks = np.array_split(entity_replace_candidates_unique, self.n_strong_candidates)

        # Initialize textB and textB_entities lists
        textB, textB_entities = [], []

        # Iterate through the n_strong_candidates chunks
        for chunk in entity_replace_candidates_chunks:

            # Set maximum number of entity_replace candidates to consider
            n_entity_replace_candidates = min(self.n_tries_entity, len(chunk))

            # Sample a random array of n_entity_replace_candidates entity_replace candidates
            entity_replace_candidates_sample = chunk[torch.randperm(n_entity_replace_candidates).numpy()]

            # Iterate through all of the entity_replace candidates
            for entity_replace_candidate in entity_replace_candidates_sample:

                # Assign candidate_head and candidate_tail
                candidate_head = headA if entity_keep == 0 else entity_replace_candidate
                candidate_tail = tailA if entity_keep == 1 else entity_replace_candidate

                # Get candidate_edges
                candidate_edges = self.graph_B.edges[candidate_head].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

                # Get all edges between entity_keep and entity_replace_candidate, according to the shuffled indices
                replace_edges = candidate_edges[candidate_edges[:, GraphDataset.TAIL_ENTITY] == candidate_tail]

                # Shuffle replace_edges
                replace_edges = replace_edges[torch.randperm(len(replace_edges)).numpy()]

                # Sample cur_textB from replace_edges
                if candidate_head == headA and candidate_tail == tailA:
                    cur_textB = self.sample_text(replace_edges, textA, 'share_two')
                else:
                    cur_textB = self.sample_text(replace_edges, textA, 'share_one', [entity_ids[entity_replace]])

                if cur_textB is not None:
                    textB.append(cur_textB) # Append cur_textB to textB list
                    textB_entities.append([candidate_head, candidate_tail])
                    break

            # Return None if textB is unsuccessfully sampled for the current chunk
            if cur_textB is None:
                return None, None, None

        return textB, textB_entities, True


    def sample_positive(self, headA, tail, textA):

        # Get edges with headA as the head
        head_edges = self.graph_B.edges[headA].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

        # Get all indices of head's neighbors, for which the neighbor is tail
        head_neighbors_idxs = np.flatnonzero(head_edges[:, GraphDataset.TAIL_ENTITY] == tail)

        # Check that head and tail are mentioned in at least one training text
        if (
            self.split == 'train' and len(head_neighbors_idxs) < 2
            or self.split == 'valid' and len(head_neighbors_idxs) < 1
        ):
            raise Exception("POSITIVE -- head and tail are not mentioned together in any training text")

        # Get all edges between head and tail
        head_tail_edges = head_edges[head_neighbors_idxs, :]

        # Shuffle head-tail edges
        head_tail_edges = head_tail_edges[torch.randperm(len(head_tail_edges)).numpy()]

        # Sample textB_pos from head-tail edges
        textB_pos = self.sample_text(head_tail_edges, textA, 'share_two')

        return textB_pos

    def sample_strong_negative(self, headA, tailA, textA):

        # If replace_tail=True, then always replace tail. Else, randomly choose replace_entity and keep_entity.
        replace_entity = np.random.randint(2)
        keep_entity = 1 - replace_entity
        entity_ids = (headA, tailA)

        # Get edges with keep_entity as the head
        keep_entity_edges = self.graph_B.edges[entity_ids[keep_entity]].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

        # Get all indices of keep_entity's neighbors, for which the neighbor is not replace_entity
        candidate_edge_idxs = np.flatnonzero(keep_entity_edges[:, GraphDataset.TAIL_ENTITY] != entity_ids[replace_entity])

        # Check that keep_entity has at least n_strong_candidates-1 neighbors besides replace_entity
        if len(candidate_edge_idxs) < self.n_strong_candidates-1:
            return None, None

        # Get all of entity_keep's edges, excluding those shared with entity_replace
        candidate_edges = keep_entity_edges[candidate_edge_idxs, :]

        # Get entity_replace candidates -- i.e., all of entity_keep's neighbors, excluding entity_replace
        entity_replace_candidates = candidate_edges[:, GraphDataset.TAIL_ENTITY]

        # Get unique array of entity_replace candidates -- i.e., all of entity_keep's neighbors, excluding entity_replace and graph duplicates
        entity_replace_candidates_unique = np.unique(entity_replace_candidates)

        # Set maximum number of replace candidates to consider
        n_entity_replace_candidates = min(self.n_tries_entity, len(entity_replace_candidates_unique))

        # Sample a random array of n_entity_replace_candidates entity_replace candidates
        entity_replace_candidates_sample = entity_replace_candidates_unique[torch.randperm(len(entity_replace_candidates_unique)).numpy()[:n_entity_replace_candidates]]

        # Initialize empty textB_strong_neg, textB_strong_neg_entities list
        textB_strong_neg, textB_strong_neg_entities = [], []

        # Iterate through all of the entity_replace candidates
        for i, entity_replace_candidate in enumerate(entity_replace_candidates_sample):
            candidate_head = headA if keep_entity == 0 else entity_replace_candidate
            candidate_tail = tailA if keep_entity == 1 else entity_replace_candidate

            # Get candidate edges, for which (head, tail) = (candidate_head, candidate_tail)
            candidate_edges = self.graph_B.edges[candidate_head].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

            # Get all edges between keep_entity and entity_replace_candidate, according to the shuffled indices
            replace_edges = candidate_edges[candidate_edges[:, GraphDataset.TAIL_ENTITY] == candidate_tail]

            # Shuffle replace_edges
            replace_edges = replace_edges[torch.randperm(len(replace_edges)).numpy()]

            # Sample cur_textB_strong_neg from replace_edges
            cur_textB_strong_neg = self.sample_text(replace_edges, textA, 'share_one', [entity_ids[replace_entity]])

            # Append cur_textB_strong_neg to textB_strong_neg list
            if cur_textB_strong_neg is not None:
                textB_strong_neg.append(cur_textB_strong_neg)
                textB_strong_neg_entities.append([candidate_head, candidate_tail])

            # Exit loop if enough textB_strong_negs have been sampled
            if len(textB_strong_neg) == self.n_strong_candidates-1:
                break
            elif i == len(entity_replace_candidates_sample)-1:
                return None, None

        return textB_strong_neg, textB_strong_neg_entities

    def sample_mtb_candidates(self, headA, tailA, textA):
        # Sample positive: textA and textB share both head and tail
        textB_pos = self.sample_positive(headA, tailA, textA)

        # Check if positive was successfully sampled
        if textB_pos is None:
            return None, None, None

        # Sample strong negatives
        textB_strong_neg, textB_strong_neg_entities = self.sample_strong_negative(headA, tailA, textA)

        # Check if strong negative was successfully sampled
        if textB_strong_neg is None:
            return None, None, None

        # Create textB and textB_entities lists
        textB = [textB_pos] + textB_strong_neg
        textB_entities = [[headA, tailA]] + textB_strong_neg_entities

        return textB, textB_entities, False

    def __getitem__(self, index):
        edge = self.graph_A[index]
        headA = edge[GraphDataset.HEAD_ENTITY].item()
        tailA = edge[GraphDataset.TAIL_ENTITY].item()

        with data_utils.numpy_seed(9031935, self.seed, self.epoch, index):
            textA = self.annotated_text_A.annotate_relation(*(edge.numpy()))

            if (
                np.max(self.similar_entities.array[headA]) == -1
                or np.max(self.similar_entities.array[tailA]) == -1
            ):
                # Sample MTB candidates
                textB, textB_entities, use_bor_candidates = self.sample_mtb_candidates(headA, tailA, textA)

            else:
                # Sample BoR candidates
                textB, textB_entities, use_bor_candidates = self.sample_bor_candidates(headA, tailA, textA)

                if textB is None:
                    # Sample MTB candidates
                    textB, textB_entities, use_bor_candidates = self.sample_mtb_candidates(headA, tailA, textA)

        # Check if textB candidates were successfully sampled
        if textB is None:
            return None

        # Make sure the correct number of textB and textB_entities have been sampled
        assert len(textB) == self.n_strong_candidates
        assert len(textB_entities) == self.n_strong_candidates

        item = {
            'textA': textA,
            'textB': textB,
            'textA_entities': [headA, tailA],
            'textB_entities': textB_entities,
            'use_bor_candidates': use_bor_candidates,
            'ntokens': len(textA),
            'nsentences': 1,
            'ntokens_AB': len(textA) + sum([len(x) for x in textB]),
        }

        return item

    def compute_candidate_weights(self, textA_entities, textB_entities, textB_strong_heads=[], textB_strong_tails=[]):

        headA, tailA = textA_entities
        headB, tailB = textB_entities[:, 0], textB_entities[:, 1]
        headA_similar_entities, tailA_similar_entities = self.similar_entities.array[headA], self.similar_entities.array[tailA]
        headA_similarity_scores, tailA_similarity_scores = self.similarity_scores.array[headA], self.similarity_scores.array[tailA]

        candidate_weights = -1 * np.ones_like(headB)
        for i in range(len(candidate_weights)):

            if headB[i] in headA_similar_entities:
                headA_headB_weight = headA_similarity_scores[np.where(headA_similar_entities == headB[i])[0][0]]
            elif headB[i] in textB_strong_heads:
                headA_headB_weight = 0.5
            else:
                headA_headB_weight = 0

            if tailB[i] in tailA_similar_entities:
                tailA_tailB_weight = tailA_similarity_scores[np.where(tailA_similar_entities == tailB[i])[0][0]]
            elif tailB[i] in textB_strong_tails:
                tailA_tailB_weight = 0.5
            else:
                tailA_tailB_weight = 0

            if tailB[i] in headA_similar_entities:
                headA_tailB_weight = self.head_tail_weight * headA_similarity_scores[np.where(headA_similar_entities == tailB[i])[0][0]]
            else:
                headA_tailB_weight = 0

            if headB[i] in tailA_similar_entities:
                tailA_headB_weight = self.head_tail_weight * tailA_similarity_scores[np.where(tailA_similar_entities == headB[i])[0][0]]
            else:
                tailA_headB_weight = 0

            candidate_weights[i] = headA_headB_weight + tailA_tailB_weight + headA_tailB_weight + tailA_headB_weight

        return F.softmax(torch.from_numpy(candidate_weights).float(), dim=0)

    def collater(self, instances):
        # Filter out instances for which no positive text pair exists
        instances = [x for x in instances if x is not None]

        # Get batch size
        batch_size = len(instances)
        if batch_size == 0:
            return None

        textA_list = []
        textB_dict = {}
        textB_entities_dict = {}
        A2B_dict = {}
        ntokens = 0
        nsentences = 0
        ntokens_AB = 0

        # Get array of headA and tailA entities (in batch idx order)
        textA_entities = np.array([instance['textA_entities'] for instance in instances])
        headA_arr = textA_entities[:, 0]
        tailA_arr = textA_entities[:, 1]

        # Get arrays of headB and tailB entities (in batch idx order)
        headB_arr = np.array(list(itertools.chain.from_iterable([instance['textB_entities'] for instance in instances])))[:, 0]
        tailB_arr = np.array(list(itertools.chain.from_iterable([instance['textB_entities'] for instance in instances])))[:, 1]

        # Get array of use_bor_candidates indicators
        use_bor_candidates = [instance['use_bor_candidates'] for instance in instances]

        # Get array of textB lengths
        textB_len = np.array(list(itertools.chain.from_iterable([[len(t) for t in instance['textB']] for instance in instances])))

        # Compute statistics for textB lengths
        textB_mean = np.mean(textB_len)
        textB_std = np.std(textB_len)

        # Generate cluster candidates based on textB lengths statistics
        bin_vals = [0] + [textB_mean + 0.5*k*textB_std for k in range(-3, 4)] + [float('inf')]
        cluster_candidates = [np.where(np.logical_and(textB_len > bin_vals[i], textB_len <= bin_vals[i+1]))[0] for i in range(len(bin_vals)-1)]

        # Build textB clusters; initialize textB_dict, textB_entities_dict, and A2B_dict
        cluster_id = 0
        textB_clusters = -1 * np.ones(batch_size * self.n_strong_candidates)
        for c in cluster_candidates:
            if len(c) > 0:
                textB_clusters[c] = cluster_id
                textB_dict[cluster_id] = []
                textB_entities_dict[cluster_id] = []
                A2B_dict[cluster_id] = []
                cluster_id += 1

        # Populate textA_list, textB_dict, and other auxiliary lists
        for i, instance in enumerate(instances):
            textA_list.append(instance['textA'])
            for j, cur_textB in enumerate(instance['textB']):
                cluster_id = textB_clusters[i * self.n_strong_candidates + j]
                textB_dict[cluster_id].append(cur_textB)
                textB_entities_dict[cluster_id].append(instance['textB_entities'][j])
                A2B_dict[cluster_id].append(i * self.n_strong_candidates + j)

            ntokens += instance['ntokens']
            nsentences += instance['nsentences']
            ntokens_AB += instance['ntokens_AB']

        # Pad textA
        padded_textA = pad_sequence(textA_list, batch_first=True, padding_value=self.dictionary.pad())

        # Pad textB, and get A2B_list
        padded_textB = {}
        padded_textB_size = 0
        textB_entities_list = []
        A2B_list = []
        for cluster_id, cluster_texts in textB_dict.items():
            padded_textB[cluster_id] = pad_sequence(cluster_texts, batch_first=True, padding_value=self.dictionary.pad())
            padded_textB_size += torch.numel(padded_textB[cluster_id])
            textB_entities_list += textB_entities_dict[cluster_id]
            A2B_list += A2B_dict[cluster_id]
        A2B_list = np.argsort(A2B_list)
        A2B = A2B_list.reshape(batch_size, self.n_strong_candidates)
        textB_entities = np.array(textB_entities_list) # in ascending length order

        # Add n weak candidates to each instance, using candidates in the current batch
        n_weak_candidates = min(self.n_weak_candidates, batch_size * self.n_strong_candidates - self.n_strong_candidates)
        textB_idxs = np.arange(batch_size * self.n_strong_candidates)
        A2B_weak_candidates = -1 * np.ones((batch_size, n_weak_candidates))
        candidate_weights = -1 * np.ones((batch_size, self.n_strong_candidates + n_weak_candidates))
        for i in range(batch_size):
            self_textB_cond = np.logical_or(textB_idxs < i*self.n_strong_candidates, textB_idxs >= i*self.n_strong_candidates+self.n_strong_candidates)

            weak_candidates_cond = np.logical_and(
                self_textB_cond,
                np.logical_not(np.logical_and(headB_arr == headA_arr[i], tailB_arr == tailA_arr[i]))
            )

            cur_weak_candidates = A2B_list[textB_idxs[weak_candidates_cond]]
            cur_weak_candidates_sample = cur_weak_candidates[torch.randperm(len(cur_weak_candidates)).numpy()][:n_weak_candidates]

            cur_weak_candidates = cur_weak_candidates_sample
            while len(cur_weak_candidates) < n_weak_candidates:
                cur_weak_candidates = np.concatenate((cur_weak_candidates, cur_weak_candidates_sample)) # pad to make up for discarded weak candidates
            A2B_weak_candidates[i, :] = cur_weak_candidates[:n_weak_candidates]

            if self.n_strong_candidates == 1:
                cur_A2B = np.concatenate(([A2B[i]], cur_weak_candidates_sample)).astype(int)
            else:
                cur_A2B = np.concatenate((A2B[i], cur_weak_candidates_sample)).astype(int)

            if use_bor_candidates[i]:
                strong_A2B = A2B_list[textB_idxs[np.logical_not(self_textB_cond)][1:]]
                candidate_weights[i, :] = self.compute_candidate_weights(
                    textA_entities[i],
                    textB_entities[cur_A2B],
                    textB_entities[strong_A2B][:, 0],
                    textB_entities[strong_A2B][:, 1]
                )
            else:
                candidate_weights[i, :] = self.compute_candidate_weights(textA_entities[i], textB_entities[cur_A2B])

        A2B = np.concatenate((A2B, A2B_weak_candidates), axis=1).flatten()

        batch_dict = {
            'textA': padded_textA,
            'textB': padded_textB,
            'A2B': torch.LongTensor(A2B),
            'target': torch.zeros(batch_size, dtype=torch.long),
            'candidate_weights': torch.FloatTensor(candidate_weights),
            'size': batch_size,
            'n_bor_instances': np.sum(use_bor_candidates),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'ntokens_AB': ntokens_AB,
            'ntokens_mem': padded_textA.numel() + padded_textB_size
        }

        return batch_dict