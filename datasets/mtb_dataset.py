from pudb import set_trace

import torch
from torch.nn.utils.rnn import pad_sequence

from copy import deepcopy
import random
import numpy as np
import numpy.random as rd

from fairseq.data import FairseqDataset

from datasets import AnnotatedTextDataset

class MTBDataset(AnnotatedTextDataset):

    def __init__(
        self,
        text_data,
        annotation_data,
        graph,
        n_entities,
        dictionary,
        max_positions,
        case0_prob,
        case1_prob,
        n_tries,
        shift_annotations,
        alpha,
    ):
        super().__init__(
            text_data,
            annotation_data,
            dictionary,
            shift_annotations,
            assign_head_tail_randomly=True,
            alpha=0.7,
        )
        self.graph = graph
        self.n_entities = n_entities
        self.max_positions = max_positions
        self.case0_prob = case0_prob
        self.case1_prob = case1_prob
        self.n_tries = n_tries

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def sample_neighbor(self, e1_neighbors, e1_edges, e2_candidates_idx=None, i=None):
        
        if e2_candidates_idx is None:
            e2_idx = random.choice(list(range(len(e1_neighbors))))
        else:
            e2_idx = e2_candidates_idx[i]

        e2 = e1_neighbors[e2_idx].item()
        e1_e2_idx = torch.nonzero(e1_neighbors == e2, as_tuple=False).flatten()
        e1_e2_edges = torch.index_select(e1_edges, 0, e1_e2_idx)

        return e2, e1_e2_edges 

    def sample_mention(self, edges, target, next_case):
        n_mention_candidates = min(self.n_tries, len(edges))
        mention_candidates_idx = torch.randperm(len(edges))[:n_mention_candidates]

        for i, m in enumerate(mention_candidates_idx):
            mention = self.insert_entity_tokens(edges[m])['mention']
            if len(mention) < self.max_positions:
                return mention, target, None
            else:
                continue

        return None, None, next_case

    def __getitem__(self, index):

        # Insert entity token into raw mention
        item = self.insert_entity_tokens(index)

        mentionA = item['mention']
        e1A = item['e1']
        e2A = item['e2']

        case = np.random.choice(3, p=[self.case0_prob, self.case1_prob, 1 - self.case0_prob - self.case1_prob])
        e1A_neighbors = self.graph[e1A]['neighbors']
        e1A_edges = self.graph[e1A]['edges']

        while True:
            # Case 0: mentionA and mentionB share both head and tail entities
            if case == 0:
                e1A_e2A_idx = torch.nonzero(e1A_neighbors == e2A, as_tuple=False).flatten()
                if len(e1A_e2A_idx) < 1:
                    #raise Exception("Case 0 -- e1A and e2A are are not mentioned in any sentence")
                    case = 1
                    continue
                elif len(e1A_e2A_idx) == 1:
                    #raise Exception("Case 0 -- e1A and e2A are only mentioned in one sentence")
                    case = 1
                    continue

                e1B = e1A
                e2B = e2A
                e1A_e2A_edges = torch.index_select(e1A_edges, 0, e1A_e2A_idx)

                mentionB, target, case = self.sample_mention(e1A_e2A_edges, 1, 1)
                
                if case is None:
                    break
                else:
                    continue

            # Case 1: mentionA and mentionB share only one entity
            elif case == 1:
                e1B = e1A
                e1B_neighbors_idx = torch.nonzero(e1A_neighbors != e2A, as_tuple=False).flatten()
                e1B_neighbors = torch.index_select(e1A_neighbors, 0, e1B_neighbors_idx)
                e1B_edges = torch.index_select(e1A_edges, 0, e1B_neighbors_idx)
                if len(e1B_neighbors) < 1:
                    #raise Exception("Case 1 -- e1A has no neighbors besides e2A")
                    case = 2
                    continue
               
                
                n_e2B_candidates = min(self.n_tries, len(e1B_neighbors))
                e2B_candidates_idx = torch.randperm(len(e1B_neighbors))[:n_e2B_candidates]
                for i in range(n_e2B_candidates):
                    e2B, e1B_e2B_edges = self.sample_neighbor(e1B_neighbors, e1B_edges, e2B_candidates_idx, i)

                    if len(e1B_e2B_edges) < 1:
                        #raise Exception("Case 1 -- No sentences mentioning e1B and e2B that do not also mention tail1")
                        continue

                    mentionB, target, case = self.sample_mention(e1B_e2B_edges, 0, 2)

                    if case is None:
                        break
                    else:
                        continue

            # Case 2: mentionA and mentionB share no entities
            else: 
                all_entities = [*range(self.n_entities)]
                all_entities.remove(e1A)
                all_entities.remove(e2A)

                while True:
                    e1B = random.choice(all_entities)
                    e1B_neighbors = self.graph[e1B]['neighbors']
                    e1B_edges = self.graph[e1B]['edges']
                    e1B_neighbors_1_idx = torch.nonzero(e1B_neighbors != e1A, as_tuple=False).flatten()
                    e1B_neighbors_2_idx = torch.nonzero(e1B_neighbors != e2A, as_tuple=False).flatten()
                    e1B_neighbors_idx = torch.unique(torch.cat((e1B_neighbors_1_idx, e1B_neighbors_2_idx)))
                    e1B_neighbors = torch.index_select(e1B_neighbors, 0, e1B_neighbors_idx)
                    e1B_edges = torch.index_select(e1B_edges, 0, e1B_neighbors_idx)

                    if len(e1B_neighbors) < 1:
                        #print("Case 2 -- e1B has no neighbors besides e1A and e2A")
                        continue

                    e2B, e1B_e2B_edges = self.sample_neighbor(e1B_neighbors, e1B_edges)
                    
                    if len(e1B_e2B_edges) < 1:
                        #print("Case 2 -- No sentences mentioning e1B and e2B that do not also mention e1A or e2A")
                        continue
                
                    mentionB, target, case = self.sample_mention(e1B_e2B_edges, 0, 2)
                    
                    if case is None:
                        break
                    else:
                        continue

                break


        return {
            'mentionA': mentionA,
            'mentionB': mentionB,
            'e1A': e1A,
            'e2A': e2A,
            'e1B': e1B,
            'e2B': e2B,
            'target':  target,
            'ntokens': max(len(mentionA), len(mentionB)),
            'nsentences': 1,
            #'ntokens': len(mention1) + len(mention2),
            #'nsentences': 2,
        }
    
    def collater(self, instances):
        if len(instances) == 0:
            return None

        mentionA_list = []
        mentionB_list = []
        e1A_list = []
        e2A_list = []
        e1B_list = []
        e2B_list = []
        target_list = []
        ntokens, nsentences = 0, 0

        for instance in instances:

            mentionA_list.append(instance['mentionA'])
            mentionB_list.append(instance['mentionB'])
            e1A_list.append(instance['e1A'])
            e2A_list.append(instance['e2A'])
            e1B_list.append(instance['e1B'])
            e2B_list.append(instance['e2B'])
            target_list.append(instance['target'])
            ntokens += instance['ntokens']
            nsentences += instance['nsentences']

        padded_mentionA = pad_sequence(mentionA_list, batch_first=True, padding_value=self.dictionary.pad())
        padded_mentionB = pad_sequence(mentionB_list, batch_first=True, padding_value=self.dictionary.pad())

        return {
            'mentionA': padded_mentionA,
            'mentionB': padded_mentionB,
            'e1A': torch.LongTensor(e1A_list),
            'e2A': torch.LongTensor(e2A_list),
            'e1B': torch.LongTensor(e1B_list),
            'e2B': torch.LongTensor(e2B_list),
            'target': torch.LongTensor(target_list),
            'size': len(instances),
            'ntokens': ntokens,
            'nsentences': nsentences,
        }
