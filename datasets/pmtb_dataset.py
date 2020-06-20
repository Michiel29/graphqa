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
        seed,
        dictionary,
        k_weak_negs,
        n_tries_entity,
        use_strong_negs
    ):
        self.split = split
        self.annotated_text_A = annotated_text_A
        self.annotated_text_B = annotated_text_B
        self.graph_A = graph_A
        self.graph_B = graph_B

        self.seed = seed
        self.dictionary = dictionary

        self.k_weak_negs = k_weak_negs
        self.n_tries_entity = n_tries_entity
        self.use_strong_negs = use_strong_negs

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

    def sample_text(self, ent_fix_new_B_edges, textA, ent_replace_A, strong_neg=False, ent_fix_A=None):

        # Iterate through edges between headB and tailB (i.e., textB candidates)
        for edge in ent_fix_new_B_edges:

            # Discard the current edge if it contains tailA
            edge_entities = self.get_edge_entities(
                self.annotated_text_B.annotation_data.array, 
                edge[GraphDataset.START_BLOCK], 
                edge[GraphDataset.END_BLOCK]
            )
            if ent_replace_A in edge_entities:
                continue

            if strong_neg and ent_fix_A in edge_entities:
                continue

            # Get textB, using the given edge, headB, and tailB
            textB = self.annotated_text_B.annotate(*(edge))

            # Check that textA and textB are not the same (this may occur for positive pairs).
            # If not, return textB.
            if not torch.equal(textA, textB):
                return textB

        # Generally, there should always be candidates satisfying both case0 and cashead.
        # We only move on to the next case if all of these candidates are longer than max_positions.
        return None

    def sample_positive(self, headA, tailA, textA):
        # base relation: (ent_fix_A, ent_replace_A)
        # positive relation: (ent_fix_B, ent_new_B)

        # Decide which target entity to replace in textA
        # head_tail_choice1 = np.random.choice(2)
        head_tail_choice1 = 0
        # if head_tail_choice1 == 0:
        #     ent_fix_A, ent_replace_A = headA, tailA
        # else:
        #     ent_fix_A, ent_replace_A = tailA, headA
        (ent_fix_A, ent_replace_A) = (headA, tailA) if head_tail_choice1 == 0 else (tailA, headA)

        # Decide whether ent_new_B is the head or tail of the sampled positive relation
        # head_tail_choice2 = np.random.choice(2)
        head_tail_choice2 = 0
        # if head_tail_choice2 == 0:
        #     ent_new_B_type = GraphDataset.TAIL_ENTITY
        # else:
        #     ent_new_B_type = GraphDataset.HEAD_ENTITY
        ent_new_B_type = GraphDataset.TAIL_ENTITY if head_tail_choice2 == 0 else GraphDataset.HEAD_ENTITY

        # Get edges for ent_fix_A
        fix_A_edges = self.graph_B.edges[ent_fix_A].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

        # Get all indices of ent_fix_A's neighbors, for which the neighbor is not tail
        fix_A_neighbors_idxs = np.flatnonzero(fix_A_edges[:, ent_new_B_type] != ent_replace_A)

        # Check that ent_fix_A has at least one neighbor besides ent_replace_A
        if len(fix_A_neighbors_idxs) < 1:
            raise Exception("POSITIVE -- ent_fix_A has no neighbors besides ent_replace_A")

        # Get all of ent_fix_B's edges, excluding those shared with ent_replace_A
        ent_fix_B = ent_fix_A
        ent_fix_B_edges = fix_A_edges[fix_A_neighbors_idxs, :]

        # Get ent_new_B candidates -- i.e., all of ent_fix_B's neighbors, excluding ent_replace_A
        ent_new_B_candidates = ent_fix_B_edges[:, ent_new_B_type]

        # Get unique array of ent_new_B candidates -- i.e., all of ent_fix_B's neighbors, excluding ent_replace_A and graph duplicates
        ent_new_B_candidates_unique = np.unique(ent_new_B_candidates)

        # Set maximum number of ent_new_B candidates to consider
        n_ent_new_B_candidates = min(self.n_tries_entity, len(ent_new_B_candidates_unique))

        # Sample a random array of n_ent_new_B_candidates ent_new_B candidates
        ent_new_B_candidates_sample = ent_new_B_candidates_unique[torch.randperm(len(ent_new_B_candidates_unique)).numpy()[:n_ent_new_B_candidates]]

        # Iterate through all of the ent_new_B candidates
        for ent_new_B in ent_new_B_candidates_sample:

            # Get all edges between ent_fix_B and ent_new_B, according to the shuffled indices
            ent_fix_new_B_edges = ent_fix_B_edges[ent_fix_B_edges[:, ent_new_B_type] == ent_new_B]

            # Shuffle ent_fix_new_B_edges
            ent_fix_new_B_edges = ent_fix_new_B_edges[torch.randperm(len(ent_fix_new_B_edges)).numpy()]

            # Sample textB_pos from ent_fix_new_B_edges
            textB_pos = self.sample_text(ent_fix_new_B_edges, textA, ent_replace_A)
            if textB_pos is not None:
                break

        return textB_pos, ent_new_B

    def sample_strong_negative(self, headA, tailA, textA, ent_new_B_pos):
        # Get edges with tailB_pos (i.e., headB_strong_neg) as the head
        tailB_pos_edges = self.graph_B.edges[tailB_pos].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

        # Get neighbors of headA
        headA_neighbors = np.unique(headA_edges[:, GraphDataset.TAIL_ENTITY])

        # Remove tailA and tailB_pos from list of headA's neighbors
        headA_neighbors = headA_neighbors[np.logical_and(headA_neighbors != tailA, headA_neighbors != tailB_pos)]

        # Get all indices of tailB_pos's neighbors, for which the neighbor is one of headA's neighbors but is not tailA/tailB_pos
        tailB_pos_neighbors_idxs = np.flatnonzero(tailB_pos_edges[:, GraphDataset.TAIL_ENTITY] in headA_neighbors)

        # Check that tailB_pos has at least one neighbor besides headA/tailA/tailB_pos
        if len(tailB_pos_neighbors_idxs) < 1:
            return None

        # Get all of headB_strong_neg's edges (note that headB_strong_neg = tailB_pos), excluding those shared with headA/tailA/tailB_pos
        headB_strong_neg_edges = tailB_pos_edges[tailB_pos_neighbors_idxs, :]

        # Get tailB_strong_neg candidates -- i.e., all of headB_strong_neg's neighbors, excluding headA/tailA/tailB_pos
        tailB_strong_neg_candidates = headB_strong_neg_edges[:, GraphDataset.TAIL_ENTITY]

        # Get unique array of tailB_strong_neg candidates -- i.e., all of headB_strong_neg's neighbors, excluding headA/tailA/tailB_pos and graph duplicates
        tailB_strong_neg_candidates_unique = np.unique(tailB_strong_neg_candidates)

        # Set maximum number of tailB_strong_neg candidates to consider
        n_tailB_strong_neg_candidates = min(self.n_tries_entity, len(tailB_strong_neg_candidates_unique))

        # Sample a random array of n_tailB_strong_neg_candidates tailB_strong_neg candidates
        tailB_strong_neg_candidates_sample = tailB_strong_neg_candidates_unique[torch.randperm(len(tailB_strong_neg_candidates_unique)).numpy()[:n_tailB_strong_neg_candidates]]

        # Iterate through all of the tailB_strong_neg candidates
        for tailB_strong_neg in tailB_strong_neg_candidates_sample:

            # Get all edges between headB_strong_neg and tailB_strong_neg, according to the shuffled indices
            headB_tailB_strong_neg_edges = headB_strong_neg_edges[headB_strong_neg_edges[:, GraphDataset.TAIL_ENTITY] == tailB_strong_neg]

            # Shuffle headB_tailB_strong_neg_edges
            headB_tailB_strong_neg_edges = headB_tailB_strong_neg_edges[torch.randperm(len(headB_tailB_strong_neg_edges)).numpy()]

            # Sample textB_strong_neg from headB_tailB_strong_neg_edges
            textB_strong_neg = self.sample_text(headB_tailB_strong_neg_edges, textA, tailA, True, headA)
            if textB_strong_neg is not None:
                break

        return textB_strong_neg

    def __getitem__(self, index):
        edge = self.graph_A[index]
        headA = edge[GraphDataset.HEAD_ENTITY].item()
        tailA = edge[GraphDataset.TAIL_ENTITY].item()

        with data_utils.numpy_seed(9031935, self.seed, self.epoch, index):
            textA = self.annotated_text_A.annotate(*(edge.numpy()))

            # Get edges with headA as the head
            # headA_edges = self.graph_B.edges[headA].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

            # Sample positive text pair: textA and textB share one target entity
            textB_pos, ent_new_B_pos = self.sample_positive(headA, tailA, textA)

            # Check if positive text pair was successfully sampled
            if textB_pos is None:
                return None

            if self.use_strong_negs:
                # Sample one strong negative text pair
                textB_strong_neg = self.sample_strong_negative(headA, tailA, textA, ent_new_B_pos)

                # Check if strong negative text pair was successfully sampled
                if textB_strong_neg is None:
                    textB = [textB_pos]
                else:
                    textB = [textB_pos, textB_strong_neg]
                
            else:
                textB = [textB_pos]

        item = {
            'textA': textA,
            'textB': textB,
            'headA': headA,
            'tailA': tailA,
            'ntokens': len(textA),
            'nsentences': 1,
            'ntokens_AB': len(textA) + len(textB_pos),
        }

        return item

    def collater(self, instances):
        # Filter out instances for which no positive text pair exists
        instances = [x for x in instances if x is not None]

        # Get batch size
        batch_size = len(instances)
        if batch_size == 0:
            return None

        textA_list = []
        textB_dict = {}
        A2B_dict = {}
        ntokens = 0
        nsentences = 0
        ntokens_AB = 0

        # Get headA and tailA lists
        headA_list = np.array([instance['headA'] for instance in instances])
        tailA_list = np.array([instance['tailA'] for instance in instances])

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
        textB_clusters = -1 * np.ones(batch_size)
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
                cluster_id = textB_clusters[i+j]
                textB_dict[cluster_id].append(cur_textB)
                A2B_dict[cluster_id].append(i+j)
            
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
        A2B_list = np.expand_dims(A2B_list, axis=1)

        # Add k weak negatives (i.e., negatives not guaranteed to be strong)
        # to each positive, using texts in the current batch
        k_weak_negs = min(self.k_weak_negs, batch_size-1)
        A2B_weak_negs = -1 * np.ones((batch_size, k_weak_negs))
        bad_weak_negs = 0
        for i in range(batch_size):
            weak_neg_conditions = np.logical_and(
                np.logical_not(np.logical_xor(headA_list == headA_list[i], tailA_list == tailA_list[i])),
                np.logical_not(np.logical_xor(headA_list == tailA_list[i], tailA_list == headA_list[i]))
            )
            weak_neg_candidates = np.flatnonzero(weak_neg_conditions)
            cur_bad_weak_negs = batch_size - len(weak_neg_candidates)
            bad_weak_negs += cur_bad_weak_negs
            weak_negs = weak_neg_candidates[torch.randperm(len(weak_neg_candidates)).numpy()]
            weak_negs = np.concatenate((weak_negs, weak_negs[:cur_bad_weak_negs])) # pad to make up for discarded weak negs
            weak_negs = weak_negs[:k_weak_negs]
            A2B_weak_negs[i, :] = weak_negs
        A2B_list = np.concatenate((A2B_list, A2B_weak_negs), axis=1).flatten()

        batch_dict = {
            'textA': padded_textA,
            'textB': padded_textB,
            'A2B': torch.LongTensor(A2B_list),
            'target': torch.zeros(batch_size, dtype=torch.long),
            'size': batch_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'ntokens_AB': ntokens_AB,
            'ntokens_mem': padded_textA.numel() + padded_textB_size,
            'bad_weak_negs': bad_weak_negs
        }

        return batch_dict