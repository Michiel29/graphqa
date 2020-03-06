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
        case0_prob,
        case1_prob,
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
        self.case0_prob = case0_prob
        self.case1_prob = case1_prob
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

    def sample_text(self, edges, e1B, e2B, target, next_case):
        n_textB_candidates = min(self.n_tries_text, len(edges))
        textB_candidates_idx = torch.randperm(len(edges))[:n_textB_candidates]

        for i, m in enumerate(textB_candidates_idx):
            textB = self.train_dataset.__getitem__(edges[m], head_entity=e1B, tail_entity=e2B)['text']
            if len(textB) <= self.max_positions:
                return textB, target, None

        # Generally, there should always be candidates satisfying case0 and case1. 
        # We only move on to the next case if all of these candidates are longer than max_positions.
        return None, None, next_case

    def __getitem__(self, index):
        item = self.split_dataset[index]

        textA = item['text']
        e1A = item['head']
        e2A = item['tail']

        case = rd.multinomial(1, [self.case0_prob, self.case1_prob, 1-self.case0_prob-self.case1_prob]).argmax()

        e1A_neighbors = self.graph[e1A]['neighbors'].numpy()
        e1A_edges = self.graph[e1A]['edges'].numpy()

        while case is not None:
            # Case 0: textA and textB share both e1 and e2
            if case == 0:

                # Set e1B and e2B to be e1A and e2A, respectively
                e1B, e2B = e1A, e2A

                # Get all indices of e1A_neighbors, for which the neighbor is e2A
                e1A_e2A_idx = np.flatnonzero(e1A_neighbors == e2A)
                
                # e1A and e2A are not mentioned in any training text
                if len(e1A_e2A_idx) < 1:
                    raise Exception("Case 0 -- e1A and e2A are not mentioned in any training text")
                
                # e1A and e2A are mentioned in only one training text
                elif len(e1A_e2A_idx) == 1:
                    raise Exception("Case 0 -- e1A and e2A are mentioned in only one training text")

                # Get all edges between e1A and e2A
                e1A_e2A_edges = np.take(e1A_edges, e1A_e2A_idx, 0)

                # Sample textB from e1A_e2A_edges
                textB, target, case = self.sample_text(e1A_e2A_edges, e1B, e2B, 1, 1)

            # Case 1: textA and textB share only e1
            elif case == 1:
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

                    # No sentences texting e1B and e2B that do not also text e1A
                    if len(e1B_e2B_edges) == 0 and i == n_e2B_candidates-1:
                        #raise Exception("Case 1 -- No sentences texting e1B and e2B that do not also text e1A")
                        case = 2
                        continue
                    elif len(e1B_e2B_edges) == 0:
                        continue

                    # Sample textB from e1B_e2B_edges
                    textB, target, case = self.sample_text(e1B_e2B_edges, e1B, e2B, 0, 2)

                    if case is None:
                        break

            # Case 2: textA and textB share no entities
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
                    textB = self.train_dataset.__getitem__(textB_idx, e1B, e2B)['text']

                    # Check that textB is not longer than max_positions
                    if len(textB) > self.max_positions:
                        continue
                    else:
                        target = 0
                        case = None
                        break

            
        return {
            'textA': textA,
            'textB': textB,
            'e1A': e1A,
            'e2A': e2A,
            'e1B': e1B,
            'e2B': e2B,
            'target': target,
            'ntokens': len(textA),
            'nsentences': 1,
            'ntokens_AB': len(textA) + len(textB),
            'textB_len': len(textB)
        }

    def collater(self, instances):
        if len(instances) == 0:
            return None

        textA_list = []
        textB_dict = {}
        e1A_list = []
        e2A_list = []
        e1B_list = []
        e2B_list = []
        target_list = []
        ntokens = 0
        nsentences = 0
        ntokens_AB = 0

        textB_len_list = [instance['textB_len'] for instance in instances]
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
                cluster_id += 1

        for i, instance in enumerate(instances):
            textA_list.append(instance['textA'])
            for cluster_id, cluster_instance_ids in textB_clusters.items():
                if i in cluster_instance_ids:
                    textB_dict[cluster_id].append(instance['textB'])
                    break

            e1A_list.append(instance['e1A'])
            e2A_list.append(instance['e2A'])
            e1B_list.append(instance['e1B'])
            e2B_list.append(instance['e2B'])
            target_list.append(instance['target'])
            ntokens += instance['ntokens']
            nsentences += instance['nsentences']
            ntokens_AB += instance['ntokens_AB']

        padded_textA = pad_sequence(textA_list, batch_first=True, padding_value=self.dictionary.pad())
        padded_textB = {}
        padded_textB_size = 0
        for cluster_id, cluster_texts in textB_dict.items():
            padded_textB[cluster_id] = pad_sequence(cluster_texts, batch_first=True, padding_value=self.dictionary.pad())
            padded_textB_size += torch.numel(padded_textB[cluster_id])

        return {
            'textA': padded_textA,
            'textB': padded_textB,
            'textB_size': padded_textB_size,
            'e1A': torch.LongTensor(e1A_list),
            'e2A': torch.LongTensor(e2A_list),
            'e1B': torch.LongTensor(e1B_list),
            'e2B': torch.LongTensor(e2B_list),
            'target': torch.LongTensor(target_list),
            'size': len(instances),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'ntokens_AB': ntokens_AB,
        }
