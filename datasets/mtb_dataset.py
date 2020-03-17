import torch
from torch.nn.utils.rnn import pad_sequence
import logging

import os
from copy import deepcopy
import numpy as np
import numpy.random as rd
import itertools

from fairseq.data import data_utils, FairseqDataset
from utils.diagnostic_utils import Diagnostic

logger = logging.getLogger(__name__)


class MTBDataset(FairseqDataset):

    def __init__(
        self,
        split_dataset,
        train_dataset,
        graph,
        n_entities,
        dictionary,
        entity_dictionary,
        k_weak_neg,
        n_tries_entity,
        n_tries_text,
        max_positions,
        seed,
        run_batch_diag=False,
    ):
        self.split_dataset = split_dataset
        self.train_dataset = train_dataset
        self.graph = graph
        self.n_texts = len(train_dataset)
        self.n_entities = n_entities
        self.dictionary = dictionary
        self.entity_dictionary = entity_dictionary
        self.k_weak_neg = k_weak_neg
        self.n_tries_entity = n_tries_entity
        self.n_tries_text = n_tries_text
        self.max_positions = max_positions
        self.seed = seed
        self.run_batch_diag = run_batch_diag
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.split_dataset)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        """Sorts by sentence length, randomly shuffled within sentences of same length"""
        return np.lexsort([
            rd.permutation(len(self)),
            self.sizes,
        ])

    @property
    def sizes(self):
        return self.split_dataset.sizes

    def sample_text(self, headB_tailB_edges, headB, tailB, textA, strong_neg=False, tailA=None):

        # Iterate through edges between headB and tailB (i.e., textB candidates) 
        for edge in headB_tailB_edges:

            # For strong negatives, discard the current edge if it contains tailA
            if strong_neg:
                edge_entity_ids = self.train_dataset.annotation_data[edge][2::3].numpy()
                if tailA in edge_entity_ids: 
                    continue
            
            # Get textB, using the given edge, headB, and tailB
            textB = self.train_dataset.__getitem__(edge, head_entity=headB, tail_entity=tailB)['text']

            # Discard textB if it is longer than max_positions
            if len(textB) > self.max_positions:
                continue
            
            # Check that textA and textB are not the same (this may occur for positive pairs). 
            # If not, return textB. 
            if not torch.equal(textA, textB): 
                return textB            

        # Generally, there should always be candidates satisfying both case0 and cashead. 
        # We only move on to the next case if all of these candidates are longer than max_positions.
        return None

    def sample_positive(self, head, tail, head_neighbors, head_edges, textA):
        # Get all indices of head_neighbors, for which the neighbor is tail
        head_tail_edges_idxs = np.flatnonzero(head_neighbors == tail)
        
        # Check that head and tail are mentioned in at least two training texts
        if len(head_tail_edges_idxs) < 1:
            raise Exception("POSITIVE -- head and tail are not mentioned together in any training text")
        elif len(head_tail_edges_idxs) == 1:
            raise Exception("POSITIVE -- head and tail are mentioned together in only one training text")

        # Get all edges between head and tail
        head_tail_edges = np.take(head_edges, head_tail_edges_idxs, 0)

        # Shuffle head-tail edges
        head_tail_edges = head_tail_edges[torch.randperm(len(head_tail_edges)).numpy()]

        # Sample textB_pos from head-tail edges
        textB_pos = self.sample_text(head_tail_edges, head, tail, textA)

        return textB_pos

    def sample_strong_negative(self, headA, tailA, headA_neighbors, headA_edges, textA):
        # Set headB to be headA
        headB = headA

        # Get tailB candidate indices -- i.e., indices of headA_neighbors, for which the neighbor is not tailA
        tailB_candidates_idxs = np.flatnonzero(headA_neighbors != tailA)
            
        # Check that headA has at least one neighbor besides tailA
        if len(tailB_candidates_idxs) == 0:
            raise Exception("STRONG NEGATIVE -- headA has no neighbors besides tailA")
        
        # Get tailB candidates -- i.e., all of headB's neighbors, excluding tailA
        tailB_candidates = np.take(headA_neighbors, tailB_candidates_idxs, 0)

        # Get unique array of tailB candidates -- i.e., all of headB's neighbors, excluding tailA and graph duplicates
        tailB_candidates_unique = np.unique(tailB_candidates)
        
        # Get all of headB's edges, excluding those shared with tailA
        headB_edges = np.take(headA_edges, tailB_candidates_idxs, 0)

        # Set maximum number of tailB candidates to consider
        n_tailB_candidates = min(self.n_tries_entity, len(tailB_candidates_unique))

        # Sample a random array of n_tailB_candidates tailB candidates
        tailB_candidates_sample = tailB_candidates_unique[torch.randperm(len(tailB_candidates_unique)).numpy()[:n_tailB_candidates]]

        # Iterate through all of the tailB candidates
        for tailB in tailB_candidates_sample:
            # Get indices of tailB_candidates corresponding to tailB
            headB_tailB_edges_idxs = np.flatnonzero(tailB_candidates == tailB)

            # Shuffle headB-tailB edge indices
            headB_tailB_edges_idxs = headB_tailB_edges_idxs[torch.randperm(len(headB_tailB_edges_idxs)).numpy()]

            # Get all edges between headB and tailB, according to the shuffled indices
            headB_tailB_edges = np.take(headB_edges, headB_tailB_edges_idxs, 0)

            # Sample textB from headB_tailB_edges
            textB = self.sample_text(headB_tailB_edges, headB, tailB, textA, strong_neg=True, tailA=tailA)
            if textB is not None:
                break

        return textB, tailB

    def sample_weak_negative(self, headA, tailA, headA_neighbors, headA_edges, textA, increment):
        textB_list, headB_list, tailB_list, textB_len_list = [], [], [], []
        while len(textB_list) < self.k_weak_neg + increment:
            # Sample an index for textB from the list of all texts
            textB_idx = rd.randint(self.n_texts)

            # Get array of unique entity ids for textB
            unique_entity_ids = np.unique(self.train_dataset.annotation_data[textB_idx][2::3].numpy())
            
            # Check that there are at least two entities in textB
            assert len(unique_entity_ids) >= 2

            # Check that headA and tailA are not in the textB 
            if headA in unique_entity_ids or tailA in unique_entity_ids:
                continue

            # Sample two of the entities in textB to use as headB and tailB
            headB, tailB = np.random.choice(unique_entity_ids, size=2, replace=False)

            # Retrieve textB token sequence, with headB and tailB marked 
            textB = self.train_dataset.__getitem__(textB_idx, headB, tailB)['text']

            # Check that textB is not longer than max_positions
            if len(textB) > self.max_positions:
                continue
            else:
                textB_list.append(textB)
                headB_list.append(headB)
                tailB_list.append(tailB)
                textB_len_list.append(len(textB))

        return textB_list, headB_list, tailB_list, textB_len_list

    def __getitem__(self, index):
        item = self.split_dataset[index]

        textA = item['text']
        headA = item['head']
        tailA = item['tail']

        with data_utils.numpy_seed(9031935, self.seed, self.epoch, index):

            # Get neighbors and edges for headA
            headA_neighbors = self.graph[headA]['neighbors'].numpy()
            headA_edges = self.graph[headA]['edges'].numpy()

            # Sample positive text pair: textA and textB share both head and tail
            textB_pos = self.sample_positive(headA, tailA, headA_neighbors, headA_edges, textA)

            # Check if positive text pair was successfully sampled
            if textB_pos is None:
                return None

            # Initialize lists for storing text pairs and their corresponding head/tail entities
            textB, headB, tailB, textB_len = [textB_pos], [headA], [tailA], [len(textB_pos)]

            # Sample one strong negative text pair
            textB_strong_neg, tailB_strong_neg = self.sample_strong_negative(headA, tailA, headA_neighbors, headA_edges, textA)

             # If we successfully sample a strong negative, then append textB_neg, headB_neg, and tailB_neg to their respective lists.
             # Otherwise, sample one extra weak negative.
            if textB_strong_neg is not None:
                textB.append(textB_strong_neg)
                headB.append(headA)
                tailB.append(tailB_strong_neg)
                textB_len.append(len(textB_strong_neg))
                weak_increment = 0
            else:
                weak_increment = 1

            # Sample [k + weak_increment] weak negative text pairs
            textB_weak_neg, headB_weak_neg, tailB_weak_neg, textB_weak_neg_len = self.sample_weak_negative(headA, tailA, headA_neighbors, headA_edges, textA, weak_increment)
            textB += textB_weak_neg
            headB += headB_weak_neg
            tailB += tailB_weak_neg
            textB_len += textB_weak_neg_len

        # A_dict = {'textA': textA, 'headA': headA, 'tailA': tailA}
        # B_dict = {'textB': textB, 'headB': headB, 'tailB': tailB}
        # diag = Diagnostic(self.dictionary, self.entity_dictionary)
        # diag.inspect_mtb_pairs(A_dict, B_dict)

        item_dict = {
            'textA': textA,
            'textB': textB,
            'textB_len': textB_len,
            'ntokens': len(textA),
            'nsentences': 1,
            'ntokens_AB': len(textA) + sum(textB_len),
        }

        if self.run_batch_diag:
            item_dict['headA'] = headA
            item_dict['tailA'] = tailA
            item_dict['headB'] = headB
            item_dict['tailB'] = tailB
        
        return item_dict

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
        target_dict = {}
        ntokens = 0
        nsentences = 0
        ntokens_AB = 0

        if self.run_batch_diag:
            headA_list = []
            tailA_list = []
            headB_list = []
            tailB_list = []
        
        # Get array of textB lengths
        textB_len = np.array([instance['textB_len'] for instance in instances])

        # Compute statistics for textB lengths
        textB_mean = np.mean(textB_len)
        textB_std = np.std(textB_len)

        # Generate cluster candidates based on textB lengths statistics
        bin_vals = [0] + [textB_mean + 0.5*k*textB_std for k in range(-3, 4)] + [float('inf')]
        cluster_candidates = [np.where(np.logical_and(textB_len > bin_vals[i], textB_len <= bin_vals[i+1])) for i in range(len(bin_vals)-1)]            
        
        # Build textB clusters; initialize textB_dict, A2B_dict, and target_dict
        cluster_id = 0
        textB_clusters = -1 * np.ones((batch_size, self.k_weak_neg+2))
        for c in cluster_candidates:
            if len(c[0]) > 0:
                textB_clusters[c[0], c[1]] = cluster_id
                textB_dict[cluster_id] = []
                A2B_dict[cluster_id] = []
                target_dict[cluster_id] = []
                cluster_id += 1

        # Populate textA_list, textB_dict, target_dict, and other auxiliary lists
        for i, instance in enumerate(instances):
            textA_list.append(instance['textA'])

            for j, cur_textB in enumerate(instance['textB']):
                cluster_id = textB_clusters[i, j]
                textB_dict[cluster_id].append(cur_textB)
                A2B_dict[cluster_id].append(i * (self.k_weak_neg+2) + j)
                if j == 0:
                    target_dict[cluster_id].append(1)
                else:
                    target_dict[cluster_id].append(0)

            if self.run_batch_diag:
                headA_list.append(instance['headA'])
                tailA_list.append(instance['tailA'])
                headB_list.append(instance['headB'])
                tailB_list.append(instance['tailB'])

            ntokens += instance['ntokens']
            nsentences += instance['nsentences']
            ntokens_AB += instance['ntokens_AB']

        # Pad textA
        padded_textA = pad_sequence(textA_list, batch_first=True, padding_value=self.dictionary.pad())

        # Pad textB; get A2B_list and target_list
        padded_textB = {}
        padded_textB_size = 0
        A2B_list = []
        target_list = []            
        for cluster_id, cluster_texts in textB_dict.items():
            padded_textB[cluster_id] = pad_sequence(cluster_texts, batch_first=True, padding_value=self.dictionary.pad())
            padded_textB_size += torch.numel(padded_textB[cluster_id])
            A2B_list += A2B_dict[cluster_id]
            target_list += target_dict[cluster_id]

        batch_dict = {
            'textA': padded_textA,
            'textB': padded_textB,
            'textB_size': padded_textB_size,
            'A2B': torch.LongTensor(np.argsort(A2B_list)),
            'target': torch.LongTensor(target_list),
            'size': batch_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'ntokens_AB': ntokens_AB,
        }

        if self.run_batch_diag:
            batch_dict['headA'] = torch.LongTensor(headA_list)
            batch_dict['tailA'] = torch.LongTensor(tailA_list)
            batch_dict['headB'] = torch.LongTensor(headB_list)
            batch_dict['tailB'] = torch.LongTensor(tailB_list)

        return batch_dict