import torch
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


class GraphDistanceDataset(FairseqDataset):

    def __init__(
        self,
        split,
        annotated_text_A,
        annotated_text_B,
        graph_A,
        graph_B,
        seed,
        dictionary,
        class_probabilities,
        n_tries_entity
    ):
        self.split = split
        self.annotated_text_A = annotated_text_A
        self.annotated_text_B = annotated_text_B
        self.graph_A = graph_A
        self.graph_B = graph_B

        self.seed = seed
        self.dictionary = dictionary

        self.class_probabilities = class_probabilities
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

    def sample_text(self, headB_tailB_edges, textA, example_class, entity_replace=None):

        # Iterate through edges between headB and tailB (i.e., textB candidates)
        for edge in headB_tailB_edges:

            # For share_one, discard the current edge if it contains entity replace
            if example_class == 'share_one':
                edge_entities = self.get_edge_entities(
                    self.annotated_text_B.annotation_data.array,
                    edge[GraphDataset.START_BLOCK],
                    edge[GraphDataset.END_BLOCK]
                )
                if entity_replace in edge_entities:
                    continue

            # Get textB, using the given edge, headB, and tailB
            textB = self.annotated_text_B.annotate(*(edge))

            # Check that textA and textB are not the same (this may occur for positive pairs).
            # If not, return textB.
            if not torch.equal(textA, textB):
                return textB

        return None

    def sample_share_both(self, head_edges, tail, textA):
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

        # Sample textB from head-tail edges
        textB = self.sample_text(head_tail_edges, textA, 'share_both')

        return textB

    def sample_share_one(self, headA, tailA, textA):
        replace_entity = np.random.randint(1)
        keep_entity = 1 - replace_entity
        entity_ids = (headA, tailA)
        entity_edge_positions = (GraphDataset.HEAD_ENTITY, GraphDataset.TAIL_ENTITY)

        keep_entity_edges = self.graph_B.edges[entity_ids[keep_entity]].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

        # Get all indices of keep_entity's neighbors, for which the neighbor is not replace_entity
        candidate_edge_idxs = np.flatnonzero(keep_entity_edges[:, entity_edge_positions[replace_entity]] != entity_ids[replace_entity])

        # Check that keep_entity has at least one neighbor besides replace_entity
        if len(candidate_edge_idxs) < 1:
            raise Exception("Keep_entity has no neighbors besides replace_entity")

        # Get all of entity_keep's edges, excluding those shared with entity_replace
        candidate_edges = keep_entity_edges[candidate_edge_idxs, :]

        # Get entity_replace candidates -- i.e., all of entity_keep's neighbors, excluding entity_replace
        entity_replace_candidates = candidate_edges[:, entity_edge_positions[replace_entity]]

        # Get unique array of entity_replace candidates -- i.e., all of entity_keep's neighbors, excluding entity_replace and graph duplicates
        entity_replace_candidates_unique = np.unique(entity_replace_candidates)

        # Set maximum number of replace candidates to consider
        n_entity_replace_candidates = min(self.n_tries_entity, len(entity_replace_candidates_unique))

        # Sample a random array of n_entity_replace_candidates entity_replace candidates
        entity_replace_candidates_sample = entity_replace_candidates_unique[torch.randperm(len(entity_replace_candidates_unique)).numpy()[:n_entity_replace_candidates]]

        # Iterate through all of the entity_replace candidates
        for entity_replace_candidate in entity_replace_candidates_sample:

            candidate_edges = self.graph_B.edges[entity_replace_candidate].numpy().reshape(-1, GraphDataset.EDGE_SIZE)
            # Get all edges between kepp_entity and entity_replace_candidate, according to the shuffled indices
            replace_edges = candidate_edges[candidate_edges[:, entity_edge_positions[keep_entity]] == entity_ids[keep_entity]]

            # Shuffle replace_edges
            replace_edges = replace_edges[torch.randperm(len(replace_edges)).numpy()]

            # Sample textB from replace_edges
            textB = self.sample_text(replace_edges, textA, 'share_one', entity_replace=entity_ids[replace_entity])
            if textB is not None:
                break

        return textB
    def sample_share_none(self, headA, tailA, textA):
        while True:
            edge = self.graph_B[np.random.randint(len(self.graph_B))].numpy()
            head = edge[GraphDataset.HEAD_ENTITY]
            tail = edge[GraphDataset.TAIL_ENTITY]
            if head != headA and tail != tailA:
                break
        edges = [edge]
        text = self.sample_text(edges, textA, 'share_none', entity_replace=None)
        return text



    def __getitem__(self, index):
        edge = self.graph_A[index].numpy()
        headA = edge[GraphDataset.HEAD_ENTITY]
        tailA = edge[GraphDataset.TAIL_ENTITY]

        with data_utils.numpy_seed(9031935, self.seed, self.epoch, index):
            textA = self.annotated_text_A.annotate(*(edge))
            example_class = np.random.choice(("share_both", "share_one", "share_none"), p=self.class_probabilities)
            textB = None
            # Get edges with headA as the head
            headA_edges = self.graph_B.edges[headA].numpy().reshape(-1, GraphDataset.EDGE_SIZE)
            if example_class == "share_both":
                # textA and textB share both head and tail
                textB = self.sample_share_both(headA_edges, tailA, textA)
                target = 0
            if example_class == "share_one" or not torch.is_tensor(textB):
                # textA and textB share either head and tail but not both
                textB = self.sample_share_one(headA, tailA, textA)
                target = 1
            if example_class == "share_none" or not torch.is_tensor(textB):
                # textA and textB share neither head nor tail
                textB = self.sample_share_none(headA, tailA, textA)
                target = 2



        item = {
            'textA': textA,
            'textB': textB,
            'target': target,
            'ntokens': len(textA),
            'nsentences': 1,
            'ntokens_AB': len(textA) + len(textB)
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
        target_list = []
        A2B_dict = {}
        ntokens = 0
        nsentences = 0
        ntokens_AB = 0

        # Get array of textB lengths
        textB_len = np.array([len(instance['textB']) for instance in instances])

        # Compute statistics for textB lengths
        textB_mean = np.mean(textB_len)
        textB_std = np.std(textB_len)

        # Generate cluster candidates based on textB lengths statistics
        bin_vals = [0] + [textB_mean + 0.5*k*textB_std for k in range(-3, 4)] + [float('inf')]
        cluster_candidates = [np.where(np.logical_and(textB_len > bin_vals[i], textB_len <= bin_vals[i+1]))[0] for i in range(len(bin_vals)-1)]

        # Build textB clusters; initialize textB_dict and A2B_dict
        cluster_id = 0
        textB_clusters = -1 * np.ones(batch_size * 2)
        for c in cluster_candidates:
            if len(c) > 0:
                textB_clusters[c] = cluster_id
                textB_dict[cluster_id] = []
                A2B_dict[cluster_id] = []
                cluster_id += 1

        # Populate textA_list, textB_dict, and other auxiliary lists
        for i, instance in enumerate(instances):
            textA_list.append(instance['textA'])
            cluster_id = textB_clusters[i]
            textB_dict[cluster_id].append(instance['textB'])
            A2B_dict[cluster_id].append(i)

            target_list.append(instance['target'])
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

        batch_dict = {
            'textA': padded_textA,
            'textB': padded_textB,
            'A2B': torch.LongTensor(A2B_list),
            'target': torch.LongTensor(target_list),
            'size': batch_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'ntokens_AB': ntokens_AB,
            'ntokens_mem': padded_textA.numel() + padded_textB_size
        }

        return batch_dict