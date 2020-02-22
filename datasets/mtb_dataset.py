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
        n_entities,
        dictionary,
        case0_prob,
        case1_prob,
        max_positions,
        n_tries,
        alpha,
        shift_annotations,
    ):
        super().__init__(
            text_data,
            annotation_data,
            dictionary,
            shift_annotations,
            assign_head_tail_randomly=True,
        )
        self.n_entities = n_entities
        self.text_data = text_data
        self.case0_prob = case0_prob
        self.case1_prob = case1_prob
        self.max_positions = max_positions
        self.n_tries = n_tries
        self.alpha = alpha

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def insert_entity_tokens(self, index):
        mention = self.text_data[index]
        annotations = self.annotation_data[index].split(3)
        
        entity_ids = self.annotation_data[index][2::3].numpy()
        unique_entity_ids = np.unique(entity_ids)
        assert len(unique_entity_ids) >= 2

        if self.assign_head_tail_randomly:
            e1_temp, e2_temp = np.random.choice(
                unique_entity_ids,
                size=2,
                replace=False,
            )

            e1_temp_indices = np.where(entity_ids == e1_temp)[0]
            e2_temp_indices = np.where(entity_ids == e2_temp)[0]

            e1_temp_idx = np.random.choice(e1_temp_indices)
            e2_temp_idx = np.random.choice(e2_temp_indices)

            if e1_temp_idx < e2_temp_idx:
                e1 = e1_temp
                e1_idx = e1_temp_idx
                e2 = e2_temp
                e2_idx = e2_temp_idx
            else:
                e1 = e2_temp
                e1_idx = e2_temp_idx
                e2 = e1_temp
                e2_idx = e1_temp_idx

        else:
            e1, e2 = unique_entity_ids[:2]
            e1_idx = 0
            e2_idx = 1

        # Get e1 and e2 start/end indices
        e1_annotation = annotations[e1_idx][2].item()
        e1_start = annotations[e1_idx][0].item() + self.shift_annotations
        e1_end = annotations[e1_idx][1].item() + self.shift_annotations

        e2_annotation = annotations[e2_idx][2].item()
        e2_start = annotations[e2_idx][0].item() + self.shift_annotations
        e2_end = annotations[e2_idx][1].item() + self.shift_annotations

        # Initialize new mention with -1's
        mention_new = -1 * torch.ones(mention.shape[0]+4).long()

        # Copy over non-entity tokens from original mention to new mention
        mention_new[:e1_start] = mention[:e1_start]
        mention_new[e1_end+2:e2_start+2] = mention[e1_end:e2_start]
        mention_new[e2_end+4:] = mention[e2_end:]

        # Insert e1 and e2 start/end tokens into new mention
        mention_new[e1_start] = self.dictionary.e1_start()
        mention_new[e1_end+1] = self.dictionary.e1_end()
        mention_new[e2_start+2] = self.dictionary.e2_start()
        mention_new[e2_end+3] = self.dictionary.e2_end()

        # For each entity, randomly decide whether to mask it with a [BLANK] token
        #   - NO, with probability alpha
        #   - YES, with probability 1-alpha
        mask_decision = np.random.choice(2, 2, p=[self.alpha, 1 - self.alpha]) 

        if mask_decision[0] == 1:
            mention_new[e1_start+1] = self.dictionary.blank() 
        else:
            mention_new[e1_start+1:e1_end+1] = mention[e1_start:e1_end]

        if mask_decision[1] == 1:
            mention_new[e2_start+3] = self.dictionary.blank()
        else:
            mention_new[e2_start+3:e2_end+3] = mention[e2_start:e2_end]

        # Remove any -1's in new mention left over after [BLANK] masking
        mention_new = mention_new[mention_new != -1] 


        return {
            'mention': mention_new,
            'e1': e1,
            'e2': e2
        }

    def sample_mention(self, edges, target, next_case):
        n_mention_candidates = min(self.n_tries, len(edges))
        mention_candidates = random.sample(edges, n_mention_candidates)

        for i, m in enumerate(mention_candidates):
            mention = self.insert_entity_tokens(m)['mention']
            if len(mention) <= self.max_positions:
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

        e1A_neighbors = self.neighbor_list[e1A]
        e1A_e2A_edges = self.edge_dict[frozenset((e1A, e2A))]

        case = np.random.choice(3, p=[self.case0_prob, self.case1_prob, 1 - self.case0_prob - self.case1_prob])

        while True:
            # Case 0: mentionA and mentionB share both head and tail entities
            if case == 0:
                if len(e1A_e2A_edges) < 1:
                    #raise Exception("Case 0 -- e1A and e2A are are not mentioned in any sentence")
                    case = 1
                    continue
                elif len(e1A_e2A_edges) == 1:
                    #raise Exception("Case 0 -- e1A and e2A are only mentioned in one sentence")
                    case = 1
                    continue

                e1B = e1A
                e2B = e2A

                mentionB, target, case = self.sample_mention(e1A_e2A_edges, 1, case+1)

                if case is None:
                    break
                else:
                    continue

            # Case 1: mentionA and mentionB share only one entity
            elif case == 1:
                e1A_neighbors_ = deepcopy(e1A_neighbors)
                del e1A_neighbors_[e2A]
                
                if len(e1A_neighbors_) < 1:
                    #raise Exception("Case 1 -- e1A has no neighbors besides e2A")
                    case = 2
                    continue
                
                e1B = e1A
                e2B = random.choice(list(e1A_neighbors_.keys()))
                e1B_e2B_edges = self.edge_dict[frozenset((e1B, e2B))]
                e1B_e2B_edges = [e for e in e1B_e2B_edges if e2A not in self.annotation_data[e][2::3]]

                if len(e1B_e2B_edges) < 1:
                    #raise Exception("Case 1 -- No sentences mentioning e1B and e2B that do not also mention tail1")
                    case = 2
                    continue

                mentionB, target, case = self.sample_mention(e1B_e2B_edges, 0, case+1)

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
                    e1B_neighbors = self.neighbor_list[e1B]
                    e1B_neighbors_ = [n for n in e1B_neighbors.keys() if (n != e1A and n != e2A)]

                    if len(e1B_neighbors_) < 1:
                        #print("Case 2 -- e1B has no neighbors besides e1A and e2A")
                        continue

                    e2B = random.choice(e1B_neighbors_)
                    e1B_e2B_edges = self.edge_dict[frozenset((e1B, e2B))]
                    e1B_e2B_edges = [e for e in e1B_e2B_edges if (e1A not in self.annotation_data[e][2::3] and e2A not in self.annotation_data[e][2::3])]

                    if len(e1B_e2B_edges) < 1:
                        #print("Case 2 -- No sentences mentioning e1B and e2B that do not also mention e1A or e2A")
                        continue
                
                    mentionB, target, case = self.sample_mention(e1B_e2B_edges, 0, case)

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
