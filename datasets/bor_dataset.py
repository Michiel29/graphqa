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
        k_weak_negs,
        n_tries_entity,
        use_strong_negs,
        replace_tail,
        mutual_neighbors,
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
        self.replace_tail = replace_tail
        self.mutual_neighbors = mutual_neighbors

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
            textB = self.annotated_text_B.annotate(*(edge))

            # Check that textA and textB are not the same (this may occur for positive pairs).
            # If not, return textB.
            if not torch.equal(textA, textB):
                return textB

        return None

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
        replace_entity = 1 if self.replace_tail else np.random.randint(2)
        keep_entity = 1 - replace_entity
        entity_ids = (headA, tailA)

        # Get edges with keep_entity as the head
        keep_entity_edges = self.graph_B.edges[entity_ids[keep_entity]].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

        if self.mutual_neighbors:
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
            return None

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

        # Iterate through all of the entity_replace candidates
        for entity_replace_candidate in entity_replace_candidates_sample:
            candidate_head = headA if keep_entity == 0 else entity_replace_candidate
            candidate_tail = tailA if keep_entity == 1 else entity_replace_candidate

            # Get candidate edges, for which (head, tail) = (candidate_head, candidate_tail)
            candidate_edges = self.graph_B.edges[candidate_head].numpy().reshape(-1, GraphDataset.EDGE_SIZE)

            # Get all edges between keep_entity and entity_replace_candidate, according to the shuffled indices
            replace_edges = candidate_edges[candidate_edges[:, GraphDataset.TAIL_ENTITY] == candidate_tail]

            # Shuffle replace_edges
            replace_edges = replace_edges[torch.randperm(len(replace_edges)).numpy()]

            # Sample textB_strong_neg from replace_edges
            textB_strong_neg = self.sample_text(replace_edges, textA, 'share_one', [entity_ids[replace_entity]])
            if textB_strong_neg is not None:
                break

        return textB_strong_neg

    def __getitem__(self, index):
        edge = self.graph_A[index]
        headA = edge[GraphDataset.HEAD_ENTITY].item()
        tailA = edge[GraphDataset.TAIL_ENTITY].item()

        with data_utils.numpy_seed(9031935, self.seed, self.epoch, index):
            textA = self.annotated_text_A.annotate(*(edge.numpy()))

            # Sample positive text pair: textA and textB share both head and tail
            textB_pos = self.sample_positive(headA, tailA, textA)

            # Check if positive text pair was successfully sampled
            if textB_pos is None:
                return None

            # Initialize textB list with textB_pos
            textB = [textB_pos]


            if self.use_strong_negs:
                # Sample one strong negative text pair
                textB_strong_neg = self.sample_strong_negative(headA, tailA, textA)

                # Check if strong negative text pair was successfully sampled
                if textB_strong_neg is None:
                    return None

                # Append textB_strong_neg to textB list
                textB.append(textB_strong_neg)


        item = {
            'textA': textA,
            'textB': textB,
            'ntokens': len(textA),
            'nsentences': 1,
            'ntokens_AB': len(textA) + sum([len(x) for x in textB]),
        }

        return item

    def collater(self, instances):
        # Filter out instances for which no positive text pair exists
        instances = [x for x in instances if x is not None]

        # Get batch size
        batch_size = len(instances)
        if batch_size == 0:
            return None

        # Get initial number of textBs per instance
        n_textB_init = 2 if self.use_strong_negs else 1

        textA_list = []
        textB_dict = {}
        A2B_dict = {}
        ntokens = 0
        nsentences = 0
        ntokens_AB = 0

        # Get array of textB lengths
        textB_len = np.array(list(itertools.chain.from_iterable([[len(t) for t in instance['textB']] for instance in instances])))

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

        # Add k weak negatives (i.e., negatives not guaranteed to be strong) to each positive,
        # using texts in the current batch
        k_weak_negs = min(self.k_weak_negs, batch_size * n_textB_init - n_textB_init)
        textB_idxs = np.arange(batch_size * n_textB_init)
        A2B_weak_negs = -1 * np.ones((batch_size, k_weak_negs))
        for i in range(batch_size):
            self_textB_cond = np.logical_and(textB_idxs != i*n_textB_init, textB_idxs != i*n_textB_init+1) if self.use_strong_negs else textB_idxs != i
            weak_neg_candidates = A2B_list[textB_idxs[self_textB_cond]]
            weak_negs = weak_neg_candidates[torch.randperm(len(weak_neg_candidates)).numpy()][:k_weak_negs]
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
            'ntokens_mem': padded_textA.numel() + padded_textB_size
        }

        return batch_dict