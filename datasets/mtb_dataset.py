import torch
from torch.nn.utils.rnn import pad_sequence
import logging

import os
from copy import deepcopy
import numpy as np
import numpy.random as rd

from fairseq.data import FairseqDataset
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
        self.entity_dictionary = entity_dictionary
        self.strong_prob = strong_prob
        self.n_tries_neighbor = n_tries_neighbor
        self.n_tries_text = n_tries_text
        self.max_positions = max_positions
        self.seed = seed
        self.epoch = 0

        self.run_batch_diag = False

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

    def sample_neighbor(self, headB_neighbors, headB_unique_neighbors, tailB_candidates_idx, i):
        tailB_idx = tailB_candidates_idx[i].item()
        tailB = headB_unique_neighbors[tailB_idx]
        headB_tailB_idx = np.flatnonzero(headB_neighbors == tailB)

        return tailB, headB_tailB_idx

    def get_edge_candidates(self, headB_edges, headB_tailB_idx, tailA, i):
        all_headB_tailB_edges = np.take(headB_edges, headB_tailB_idx, 0)
        headB_tailB_edges = []
        for i, edge in enumerate(all_headB_tailB_edges):
            edge_entity_ids = self.train_dataset.annotation_data[edge][2::3].numpy()
            if tailA not in edge_entity_ids: 
                headB_tailB_edges.append(edge)
        headB_tailB_edges = np.array(headB_tailB_edges)

        return headB_tailB_edges

    def sample_text(self, edges, headB, tailB, textA):
        n_textB_candidates = min(self.n_tries_text, len(edges))
        textB_candidates_idx = torch.randperm(len(edges))[:n_textB_candidates]

        for m in textB_candidates_idx:
            textB = self.train_dataset.__getitem__(edges[m], head_entity=headB, tail_entity=tailB)['text']
            if len(textB) > self.max_positions:
                continue
            if not torch.equal(textA, textB):
                return textB                

        # Generally, there should always be candidates satisfying both case0 and cashead. 
        # We only move on to the next case if all of these candidates are longer than max_positions.
        return None

    def sample_positive_pair(self, head, tail, head_neighbors, head_edges, textA):
        # Get all indices of head_neighbors, for which the neighbor is tail
        head_tail_idx = np.flatnonzero(head_neighbors == tail)
        
        # head and tail are not mentioned in any training text
        if len(head_tail_idx) < 1:
            raise Exception("Case 0 -- head and tail are not mentioned in any training text")
        
        # head and tail are mentioned in only one training text
        elif len(head_tail_idx) == 1:
            raise Exception("Case 0 -- head and tail are mentioned in only one training text")

        # Get all edges between head and tail
        head_tail_edges = np.take(head_edges, head_tail_idx, 0)

        # Sample textB from head_tail_edges
        textB_pos = self.sample_text(head_tail_edges, head, tail, textA)

        return textB_pos, head, tail

    def sample_negative_pair(self, headA, tailA, headA_neighbors, headA_edges, neg_type, textA): 

        found_neg_pair = False
        while not found_neg_pair:

            # Sample a strong negative: textA and textB share only head
            if neg_type == 0:
                # Set headB to be headA
                headB = headA

                # Get all indices of headA_neighbors, for which the neighbor is not headB or tailA
                headB_neighbors_idx = np.flatnonzero(np.logical_and(headA_neighbors != headB, headA_neighbors != tailA))
                 
                # headA has no neighbors besides tailA
                if len(headB_neighbors_idx) == 0:
                    raise Exception("Case 1 -- headA has no neighbors besides headB and tailA")
                
                # Get all of headB's neighbors, excluding headB and tailA
                headB_neighbors = np.take(headA_neighbors, headB_neighbors_idx, 0)

                # Get all of headB's neighbors, excluding headB, tailA, and duplicates
                headB_unique_neighbors = np.unique(headB_neighbors)
                
                # Get all of headB's edges, excluding those corresponding to headB and tailA
                headB_edges = np.take(headA_edges, headB_neighbors_idx, 0)

                # Set number of tailB candidates
                n_tailB_candidates = min(self.n_tries_neighbor, len(headB_unique_neighbors))

                # Get a random array of tailB candidates (which are indices of headB_unique_neighbors)
                tailB_candidates_idx = torch.randperm(len(headB_unique_neighbors))[:n_tailB_candidates]

                # Iterate through all of the tailB candidates
                for i in range(n_tailB_candidates):

                    # Sample tailB, and return indices of headB_neighbors corresponding to tailB
                    tailB, headB_tailB_idx = self.sample_neighbor(headB_neighbors, headB_unique_neighbors, tailB_candidates_idx, i)

                    # Get an array of edges between headB and tailB
                    headB_tailB_edges = self.get_edge_candidates(headB_edges, headB_tailB_idx, tailA, i)

                    # No sentences mentioning headB and tailB that do not also text headA
                    if len(headB_tailB_edges) == 0 and i == n_tailB_candidates-1:
                        #raise Exception("Case 1 -- No sentences mentioning headB and tailB that do not also text headA")
                        neg_type = 1
                        continue
                    elif len(headB_tailB_edges) == 0:
                        continue

                    # Sample textB from headB_tailB_edges
                    textB_neg = self.sample_text(headB_tailB_edges, headB, tailB, textA)

                    if textB_neg is not None:
                        found_neg_pair = True
                        break
                    elif i == n_tailB_candidates-1:
                        neg_type = 1
                    else:
                        continue

            # Sample a weak negative: textA and textB share no entities
            else:
                while not found_neg_pair:
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
                    textB_neg = self.train_dataset.__getitem__(textB_idx, headB, tailB)['text']

                    # Check that textB is not longer than max_positions
                    if len(textB_neg) > self.max_positions:
                        continue
                    else:
                        found_neg_pair = True


        return textB_neg, headB, tailB, neg_type


    def __getitem__(self, index):
        item = self.split_dataset[index]

        textA = item['text']
        headA = item['head']
        tailA = item['tail']

        # Sample type of negative pair: 0 (strong) or 1 (weak)
        neg_type = rd.multinomial(1, [self.strong_prob, 1-self.strong_prob]).argmax()

        # Get neighbors and edges for headA
        headA_neighbors = self.graph[headA]['neighbors'].numpy()
        headA_edges = self.graph[headA]['edges'].numpy()

        # Sample positive text pair: textA and textB share both head and tail
        textB_pos, headB_pos, tailB_pos = self.sample_positive_pair(headA, tailA, headA_neighbors, headA_edges, textA)

        # Check if positive text pair was successfully sampled
        if textB_pos is not None:
            # Sample negative text pair -- must be successful, at least for weak negatives
            textB_neg, headB_neg, tailB_neg, neg_type = self.sample_negative_pair(headA, tailA, headA_neighbors, headA_edges, neg_type, textA)
        else:
            return None

        assert headA != tailA and headB_pos != tailB_pos and headB_neg != tailB_neg # check that head and tail are different
        assert headA == headB_pos and tailA == tailB_pos # check that entities are valid for positive pair
        if neg_type == 0:
            assert headA == headB_neg and tailA != tailB_neg # check that entities are valid for strong negative pair
        else:
            assert headA != headB_neg and tailA != tailB_neg # check that entities are valid for weak negative pair

        # diag = Diagnostic(self.dictionary, self.entity_dictionary)
        # texts_dict = {'textA': textA, 'textB_pos': textB_pos, 'textB_neg': textB_neg}
        # entities_dict = {'headA': headA, 'tailA': tailA, 'headB_neg': headB_neg, 'tailB_neg': tailB_neg}
        # self.diag.inspect_mtb_pairs(texts_dict, entities_dict)

        item_dict = {
            'textA': textA,
            'textB_pos': textB_pos,
            'textB_neg': textB_neg,
            'target_pos': 1,
            'target_neg': 0,
            'ntokens': len(textA),
            'nsentences': 1,
            'ntokens_AB': len(textA) + len(textB_pos) + len(textB_neg),
            'textB_pos_len': len(textB_pos),
            'textB_neg_len': len(textB_neg),
        }

        if self.run_batch_diag:
            item_dict['headA'] = headA
            item_dict['tailA'] = tailA
            item_dict['headB_neg'] = headB_neg
            item_dict['tailB_neg'] = tailB_neg
            item_dict['neg_type'] = neg_type
        
        return item_dict



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

        if self.run_batch_diag:
            B2A_list = []
            A2B = {}
            headA_list = []
            tailA_list = []
            headB_neg_list = []
            tailB_neg_list = []
            neg_type_list = []

        textB_pos_len_list = [instance['textB_pos_len'] for instance in instances]
        textB_neg_len_list = [instance['textB_neg_len'] for instance in instances]
        textB_len_list = np.array(textB_pos_len_list + textB_neg_len_list)
        
        textB_mean = np.mean(textB_len_list)
        textB_std = np.std(textB_len_list)

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
            
            if self.run_batch_diag:
                headA_list.append(instance['headA'])
                tailA_list.append(instance['tailA'])
                headB_neg_list.append(instance['headB_neg'])
                tailB_neg_list.append(instance['tailB_neg'])
                neg_type_list.append(instance['neg_type'])

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
            if self.run_batch_diag:
                B2A_list += B2A_dict[cluster_id]

        batch_dict = {
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

        if self.run_batch_diag:
            A2B = np.lexsort((1-np.array(target_list), B2A_list))
            batch_dict['A2B'] = torch.from_numpy(A2B).long()
            batch_dict['headA'] = torch.LongTensor(headA_list)
            batch_dict['tailA'] = torch.LongTensor(tailA_list)
            batch_dict['headB_neg'] = torch.LongTensor(headB_neg_list)
            batch_dict['tailB_neg'] = torch.LongTensor(tailB_neg_list)
            batch_dict['neg_type'] = torch.LongTensor(neg_type_list)

        return batch_dict