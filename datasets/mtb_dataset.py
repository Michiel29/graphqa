import torch

import random
import numpy as np
import numpy.random as rd

from fairseq.data import FairseqDataset

from datasets import AnnotatedTextDataset

from torch.nn.utils.rnn import pad_sequence

class MTBDataset(AnnotatedTextDataset):

    def __init__(
        self,
        text_data,
        annotation_data,
        n_entities,
        dictionary,
        case0_prob,
        case1_prob,
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

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def __getitem__(self, index):

        item = super().__getitem__(index)

        mention1 = item['mention']
        head1 = item['head']
        tail1 = item['tail']

        head1_neighbors = self.neighbor_list[head1]
        head1_tail1_edges = self.edge_dict[frozenset((head1, tail1))]
        
        case = np.random.choice(3, p=[self.case0_prob, self.case1_prob, 1 - self.case0_prob - self.case1_prob])

        print('head1: {}'.format(head1))
        print('tail1: {}'.format(tail1))
        print('mention1: {}'.format(mention1))
        print('head1_tail1_edges: {}'.format(head1_tail1_edges))
        

        while True:
            # Case 0: mention_A and mention_B share both head and tail entities
            if case == 0:
                if len(head1_tail1_edges) == 0:
                    raise Exception("Case 0 -- head1 and tail1 are are not mentioned in any sentence")
                elif len(head1_tail1_edges) == 1:
                    #raise Exception("Case 0 -- head1 and tail1 are only mentioned in one sentence")
                    case = 1
                    continue

                head2 = head1
                tail2 = tail1

                mention2_idx = random.choice(head1_tail1_edges)
                mention2 = self.__getitem__(mention2_idx)['mention']
                target = 1
                break

            # Case 1: mention_A and mention_B share only one entity
            elif case == 1:
                head1_neighbors.remove(tail1)
                
                if len(head1_neighbors) == 0:
                    #raise Exception("Case 1 -- head1 has no neighbors besides tail1")
                    case = 2
                    continue
                
                head2 = head1
                tail2 = random.choice(head1_neighbors)

                head2_tail2_edges = self.edge_dict[frozenset((head2, tail2))]
                head2_tail2_edges = [e for e in head2_tail2_edges if tail1 not in self.annotation_data[e][2::3]]

                if len(head2_tail2_edges) == 0:
                    #raise Exception("Case 1 -- No sentences mentioning head2 and tail2 that do not also mention tail1")
                    case = 2
                    continue

                mention2_idx = random.choice(head2_tail2_edges)
                mention2 = self.__getitem__(mention2_idx)['mention']
                target = 0
                break

            # Case 2: mention_A and mention_B share no entities
            else: 
                all_entities = [*range(self.n_entities)]
                all_entities.remove(head1)
                all_entities.remove(tail1)

                while True:
                    head2 = random.choice(all_entities)
                    head2_neighbors = self.neighbor_list[head2]
                    head2_neighbors = [n for n in head2_neighbors if (n != head1 and n != tail1)]

                    if len(head2_neighbors) == 0:
                        #print("Case 2 -- head2 has no neighbors besides head1 and tail1")
                        continue

                    else:
                        tail2 = random.choice(head2_neighbors)
                        head2_tail2_edges = self.edge_dict[frozenset((head2, tail2))]
                        head2_tail2_edges = [e for e in head2_tail2_edges if (head1 not in self.annotation_data[e][2::3] and tail1 not in self.annotation_data[e][2::3])]

                        if len(head2_tail2_edges) == 0:
                            #print("Case 2 -- No sentences mentioning head2 and tail2 that do not also mention head1 or tail1")
                            continue
                        else:
                            break
                
                mention2_idx = random.choice(head2_tail2_edges)
                mention2 = self.__getitem__(mention2_idx)['mention']
                target = 0
                break

        return {
            'mention1': mention1,
            'mention2': mention2,
            'head1':  head1,
            'head2':  head2,
            'tail1': tail1,
            'tail2': tail2,
            'target':  target,
            'ntokens': len(mention1) + len(mention2),
            'nsentences': 2,
        }
    
    def collater(self, instances):
        if len(instances) == 0:
            return None

        mention1_list = []
        mention2_list = []
        head1_list = []
        head2_list = []
        tail1_list = []
        tail2_list = []
        target_list = []
        ntokens, nsentences = 0, 0

        for instance in instances:

            mention1_list.append(instance['mention1'])
            mention2_list.append(instance['mention2'])
            head1_list.append(instance['head1'])
            head2_list.append(instance['head2'])
            tail1_list.append(instance['tail1'])
            tail2_list.append(instance['tail2'])
            target_list.append(instance['target'])
            ntokens += instance['ntokens']
            nsentences += instance['nsentences']

        padded_mentions1 = pad_sequence(mention1_list, batch_first=True, padding_value=self.dictionary.pad())
        padded_mentions2 = pad_sequence(mention2_list, batch_first=True, padding_value=self.dictionary.pad())

        return {
            'mention1': padded_mentions1,
            'mention2': padded_mentions2,
            'head1':  torch.LongTensor(head1_list),
            'head2':  torch.LongTensor(head2_list),
            'tail1': torch.LongTensor(tail1_list),
            'tail2': torch.LongTensor(tail2_list),
            'target':  torch.LongTensor(target_list),
            'size': len(instances),
            'ntokens': ntokens,
            'nsentences': nsentences,
        }
