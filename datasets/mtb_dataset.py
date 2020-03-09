import torch
from torch.nn.utils.rnn import pad_sequence
import logging

import os
from copy import deepcopy
import numpy as np
import numpy.random as rd

from fairseq.data import FairseqDataset


logger = logging.getLogger(__name__)


class MTBDataset(FairseqDataset):

    def __init__(
        self,
        split_dataset,
        train_dataset,
        graph,
        n_entities,
        dictionary,
        strong_prob,
        n_tries_neighbor,
        n_tries_text,
        max_positions,
        seed,
    ):
        self.split_dataset = split_dataset
        self.train_dataset = train_dataset
        self.graph = graph
        self.n_texts = len(train_dataset)
        self.n_entities = n_entities
        self.dictionary = dictionary
        self.strong_prob = strong_prob
        self.n_tries_neighbor = n_tries_neighbor
        self.n_tries_text = n_tries_text
        self.max_positions = max_positions
        self.seed = seed
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

    def sample_neighbor(self, e1B_neighbors, e1B_unique_neighbors, e1B_edges, e2A, e2B_candidates_idx, i):
        e2B_idx = e2B_candidates_idx[i].item()
        e2B = e1B_unique_neighbors[e2B_idx]
        e1B_e2B_idx = np.flatnonzero(e1B_neighbors == e2B)

        all_e1B_e2B_edges = np.take(e1B_edges, e1B_e2B_idx, 0)
        e1B_e2B_edges = []
        for i, edge in enumerate(all_e1B_e2B_edges):
            edge_entity_ids = self.train_dataset.annotation_data[edge][2::3].numpy()
            if e2A not in edge_entity_ids: 
                e1B_e2B_edges.append(edge)
        e1B_e2B_edges = np.array(e1B_e2B_edges)

        return e2B, e1B_e2B_edges

    def sample_text(self, edges, e1B, e2B):
        n_textB_candidates = min(self.n_tries_text, len(edges))
        textB_candidates_idx = torch.randperm(len(edges))[:n_textB_candidates]

        for i, m in enumerate(textB_candidates_idx):
            textB = self.train_dataset.__getitem__(edges[m], head_entity=e1B, tail_entity=e2B)['text']
            if len(textB) <= self.max_positions:
                return textB

        # Generally, there should always be candidates satisfying both case0 and case1. 
        # We only move on to the next case if all of these candidates are longer than max_positions.
        return None

    def sample_positive_pair(self, e1, e2, e1_neighbors, e1_edges):
        # Get all indices of e1_neighbors, for which the neighbor is e2
        e1_e2_idx = np.flatnonzero(e1_neighbors == e2)
        
        # e1 and e2 are not mentioned in any training text
        if len(e1_e2_idx) < 1:
            raise Exception("Case 0 -- e1 and e2 are not mentioned in any training text")
        
        # e1 and e2 are mentioned in only one training text
        elif len(e1_e2_idx) == 1:
            raise Exception("Case 0 -- e1 and e2 are mentioned in only one training text")

        # Get all edges between e1 and e2
        e1_e2_edges = np.take(e1_edges, e1_e2_idx, 0)

        # Sample textB from e1_e2_edges
        textB_pos = self.sample_text(e1_e2_edges, e1, e2)

        return textB_pos

    def sample_negative_pair(self, e1A, e2A, e1A_neighbors, e1A_edges, neg_type): 

        while neg_type is not None:

            # Sample a strong negative: textA and textB share only e1
            if neg_type == 0:
                # Set e1B to be e1A
                e1B = e1A

                # Get all indices of e1A_neighbors, for which the neighbor is not e2A
                e1B_neighbors_idx = np.flatnonzero(e1A_neighbors != e2A)
                 
                # e1A has no neighbors besides e2A
                if len(e1B_neighbors_idx) == 0:
                    raise Exception("Case 1 -- e1A has no neighbors besides e2A")
                
                # Get all of e1B's neighbors, excluding e2A
                e1B_neighbors = np.take(e1A_neighbors, e1B_neighbors_idx, 0)

                # Get all of e1B's neighbors, excluding both e2A and duplicates
                e1B_unique_neighbors = np.unique(e1B_neighbors)
                
                # Get all of e1B's edges, excluding those corresponding to e2A
                e1B_edges = np.take(e1A_edges, e1B_neighbors_idx, 0)

                # Set number of e2B candidates
                n_e2B_candidates = min(self.n_tries_neighbor, len(e1B_unique_neighbors))

                # Get a random array of e2B candidates (which are indices of e1B_unique_neighbors)
                e2B_candidates_idx = torch.randperm(len(e1B_unique_neighbors))[:n_e2B_candidates]

                # Iterate through all of the e2B candidates
                for i in range(n_e2B_candidates):

                    # Sample e2B, and return an array of edges between e1B and e2B
                    e2B, e1B_e2B_edges = self.sample_neighbor(e1B_neighbors, e1B_unique_neighbors, e1B_edges, e2A, e2B_candidates_idx, i)

                    # No sentences mentioning e1B and e2B that do not also text e1A
                    if len(e1B_e2B_edges) == 0 and i == n_e2B_candidates-1:
                        #raise Exception("Case 1 -- No sentences mentioning e1B and e2B that do not also text e1A")
                        neg_type = 1
                        continue
                    elif len(e1B_e2B_edges) == 0:
                        continue

                    # Sample textB from e1B_e2B_edges
                    textB_neg = self.sample_text(e1B_e2B_edges, e1B, e2B)

                    if textB_neg is not None:
                        neg_type = None
                        break
                    else:
                        neg_type = 1

            # Sample a weak negative: textA and textB share no entities
            else:

                while True:
                    # Sample an index for textB from the list of all texts
                    textB_idx = rd.randint(self.n_texts)

                    # Get array of unique entity ids for textB
                    unique_entity_ids = np.unique(self.train_dataset.annotation_data[textB_idx][2::3].numpy())
                    
                    # Check that there are at least two entities in textB
                    assert len(unique_entity_ids) >= 2

                    # Check that e1A and e2A are not in the textB 
                    if e1A in unique_entity_ids or e2A in unique_entity_ids:
                        continue

                    # Sample two of the entities in textB to use as e1B and e2B
                    e1B, e2B = np.random.choice(unique_entity_ids, size=2, replace=False)

                    # Retrieve textB token sequence, with e1B and e2B marked 
                    textB_neg = self.train_dataset.__getitem__(textB_idx, e1B, e2B)['text']

                    # Check that textB is not longer than max_positions
                    if len(textB_neg) > self.max_positions:
                        continue
                    else:
                        neg_type = None
                        break

        return textB_neg, e1B, e2B


    def __getitem__(self, index):
        item = self.split_dataset[index]

        textA = item['text']
        e1A = item['head']
        e2A = item['tail']

        # Sample type of negative pair: 0 (strong) or 1 (weak)
        neg_type = rd.multinomial(1, [self.strong_prob, 1-self.strong_prob]).argmax()

        # Get neighbors and edges for e1A
        e1A_neighbors = self.graph[e1A]['neighbors'].numpy()
        e1A_edges = self.graph[e1A]['edges'].numpy()

        # Sample positive text pair: textA and textB share both e1 and e2
        textB_pos = self.sample_positive_pair(e1A, e2A, e1A_neighbors, e1A_edges)

        # Check if positive text pair was successfully sampled
        if textB_pos is not None:
            # Sample negative text pair -- must be successful, at least for weak negatives
            textB_neg, e1B_neg, e2B_neg = self.sample_negative_pair(e1A, e2A, e1A_neighbors, e1A_edges, neg_type)
            target_pos, target_neg = 1, 0
        else:
            return None

        return {
            'textA': textA,
            'textB_pos': textB_pos,
            'textB_neg': textB_neg,
            'target_pos': target_pos,
            'target_neg': target_neg,
            'ntokens': len(textA),
            'nsentences': 1,
            'ntokens_AB': len(textA) + len(textB_pos) + len(textB_neg),
            'textB_pos_len': len(textB_pos),
            'textB_neg_len': len(textB_neg),
        }

    def collater(self, instances):
        # Filter out instances for which no positive text pair exists 
        instances = [x for x in instances if x is not None]

        batch_size = len(instances)
        if batch_size == 0:
            return None

        textA_list = []
        textB_dict = {}
        B2A_dict = {}
        target_dict = {}
        ntokens = 0
        nsentences = 0
        ntokens_AB = 0

        textB_pos_len_list = [instance['textB_pos_len'] for instance in instances]
        textB_neg_len_list = [instance['textB_neg_len'] for instance in instances]
        textB_len_list = np.array(textB_pos_len_list + textB_neg_len_list)
        
        textB_mean = np.mean(textB_len_list)
        textB_std = np.std(textB_len_list)
        textB_max = np.max(textB_len_list)

        cluster_candidates = [
                                np.where(textB_len_list < textB_mean - 1.5*textB_std)[0], # len < mean-1.5*std
                                np.where(np.logical_and(textB_len_list >= textB_mean - 1.5*textB_std, textB_len_list < textB_mean - textB_std))[0], # mean-1.5*std <= len < mean-std
                                np.where(np.logical_and(textB_len_list >= textB_mean - textB_std, textB_len_list < textB_mean - 0.5*textB_std))[0], # mean-std <= len < mean-0.5*std
                                np.where(np.logical_and(textB_len_list >= textB_mean - 0.5*textB_std, textB_len_list < textB_mean))[0], # mean-0.5*std <= len < mean
                                np.where(np.logical_and(textB_len_list >= textB_mean, textB_len_list < textB_mean + 0.5*textB_std))[0], # mean <= len < mean+0.5*std
                                np.where(np.logical_and(textB_len_list >= textB_mean + 0.5*textB_std, textB_len_list < textB_mean + textB_std))[0], # mean+0.5*std <= len < mean+std
                                np.where(np.logical_and(textB_len_list >= textB_mean + textB_std, textB_len_list < textB_mean + 1.5*textB_std))[0], # mean+std <= len < mean+1.5*std
                                np.where(textB_len_list >= textB_mean + 1.5*textB_std)[0] # mean+1.5*std <= len
                             ]
        
        textB_clusters = {}
        cluster_id = 0
        for c in cluster_candidates:
            if len(c) > 0:
                textB_clusters[cluster_id] = c
                textB_dict[cluster_id] = []
                B2A_dict[cluster_id] = []
                target_dict[cluster_id] = []
                cluster_id += 1

        for i, instance in enumerate(instances):
            textA_list.append(instance['textA'])

            B2A_pos, B2A_neg = False, False
            for cluster_id, cluster_instance_ids in textB_clusters.items():
                if not B2A_pos and i in cluster_instance_ids[cluster_instance_ids < batch_size]:
                    textB_dict[cluster_id].append(instance['textB_pos'])
                    B2A_dict[cluster_id].append(i)
                    target_dict[cluster_id].append(instance['target_pos'])
                    B2A_pos = True
                if not B2A_neg and i in cluster_instance_ids[cluster_instance_ids >= batch_size]-batch_size:
                    textB_dict[cluster_id].append(instance['textB_neg'])
                    B2A_dict[cluster_id].append(i)
                    target_dict[cluster_id].append(instance['target_neg'])
                    B2A_neg = True
                if B2A_pos and B2A_neg:
                    break
            
            ntokens += instance['ntokens']
            nsentences += instance['nsentences']
            ntokens_AB += instance['ntokens_AB']

        padded_textA = pad_sequence(textA_list, batch_first=True, padding_value=self.dictionary.pad())
        padded_textB = {}
        padded_textB_size = 0
        target_list = []
        for cluster_id, cluster_texts in textB_dict.items():
            padded_textB[cluster_id] = pad_sequence(cluster_texts, batch_first=True, padding_value=self.dictionary.pad())
            padded_textB_size += torch.numel(padded_textB[cluster_id])
            target_list += target_dict[cluster_id]

        return {
            'textA': padded_textA,
            'textB': padded_textB,
            'textB_size': padded_textB_size,
            'B2A': B2A_dict,
            'target': torch.LongTensor(target_list),
            'size': batch_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'ntokens_AB': ntokens_AB,
        }
