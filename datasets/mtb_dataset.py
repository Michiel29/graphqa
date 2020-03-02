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
        split,
        text_data,
        annotation_data,
        graph,
        graph_text_data,
        graph_annotation_data,
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
            mask_type='start_end',
            graph_text_data=graph_text_data,
            graph_annotation_data=graph_annotation_data,
            alpha=alpha,
        )
        self.split = split
        assert split in ['train', 'valid']
        self.text_data = text_data
        self.graph = graph
        self.n_entities = n_entities
        self.max_positions = max_positions
        self.case0_prob = case0_prob
        self.case1_prob = case1_prob
        self.n_tries = n_tries
        self.all_entities = [*range(self.n_entities)]
        self.epoch = 0

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def sample_neighbor(self, e1_neighbors, e1_edges, e2_candidates_idx=None, i=None):

        if e2_candidates_idx is None:
            e2_idx = random.choice(list(range(len(e1_neighbors))))
        else:
            e2_idx = e2_candidates_idx[i].item()

        e2 = e1_neighbors[e2_idx]
        e1_e2_idx = np.flatnonzero(e1_neighbors == e2)
        e1_e2_edges = np.take(e1_edges, e1_e2_idx, 0)

        return e2, e1_e2_edges

    def sample_mention(self, edges, target, next_case):

        n_mentionB_candidates = min(self.n_tries, len(edges))
        mentionB_candidates_idx = torch.randperm(len(edges))[:n_mentionB_candidates]

        for i, m in enumerate(mentionB_candidates_idx):
            if self.split == 'train':
                mentionB = super().__getitem__(edges[m])['mention']
            else:
                mentionB = super().__getitem__(edges[m], True)['mention']

            if len(mentionB) < self.max_positions:
                return mentionB, target, None
            else:
                continue

        return None, None, next_case

    def __getitem__(self, index):

        item = super().__getitem__(index)

        mentionA = item['mention']
        e1A = item['e1']
        e2A = item['e2']

        case = rd.multinomial(1, [self.case0_prob, self.case1_prob, 1-self.case0_prob-self.case1_prob]).argmax()

        e1A_neighbors = self.graph[e1A]['neighbors'].numpy()
        e1A_edges = self.graph[e1A]['edges'].numpy()

        while case is not None:

            # Case 0: mentionA and mentionB share both head and tail entities
            if case == 0:
                e1B, e2B = e1A, e2A
                e1A_e2A_idx = np.flatnonzero(e1A_neighbors == e2A)

                if len(e1A_e2A_idx) < 1:
                    case = 1
                    continue
                    #raise Exception("Case 0 -- e1A and e2A are are not mentioned in any sentence")

                # e1A and e2A are only mentioned in one sentence
                elif len(e1A_e2A_idx) == 1:
                    case = 1
                    continue

                e1A_e2A_edges = np.take(e1A_edges, e1A_e2A_idx, 0)
                mentionB, target, case = self.sample_mention(e1A_e2A_edges, 1, 1)

            # Case 1: mentionA and mentionB share only one entity
            elif case == 1:
                e1B = e1A
                e1B_neighbors_idx = np.flatnonzero(e1A_neighbors != e2A)
                e1B_neighbors = np.take(e1A_neighbors, e1B_neighbors_idx, 0)
                e1B_edges = np.take(e1A_edges, e1B_neighbors_idx, 0)

                # e1A has no neighbors besides e2A
                if len(e1B_neighbors) < 1:
                    case = 2
                    continue

                n_e2B_candidates = min(self.n_tries, len(e1B_neighbors))
                e2B_candidates_idx = torch.randperm(len(e1B_neighbors))[:n_e2B_candidates]
                for i in range(n_e2B_candidates):
                    e2B, e1B_e2B_edges = self.sample_neighbor(e1B_neighbors, e1B_edges, e2B_candidates_idx, i)

                    # No sentences mentioning e1B and e2B that do not also mention tail1
                    if len(e1B_e2B_edges) < 1:
                        continue

                    mentionB, target, case = self.sample_mention(e1B_e2B_edges, 0, 2)

                    if case is None:
                        break

            # Case 2: mentionA and mentionB share no entities
            else:
                while True:
                    e1B = random.choice(self.all_entities)
                    if e1B not in [e1A, e2A]:
                        break

                e1B_neighbors = self.graph[e1B]['neighbors'].numpy()
                e1B_edges = self.graph[e1B]['edges'].numpy()
                e1B_neighbors_1_idx = np.flatnonzero(e1B_neighbors != e1A)
                e1B_neighbors_2_idx = np.flatnonzero(e1B_neighbors != e2A)
                e1B_neighbors_idx = np.unique(np.concatenate((e1B_neighbors_1_idx, e1B_neighbors_2_idx)))

                # e1B has no neighbors besides e1A and e2A
                if len(e1B_neighbors_idx) < 1:
                    continue

                e1B_neighbors = np.take(e1B_neighbors, e1B_neighbors_idx, 0)
                e1B_edges = np.take(e1B_edges, e1B_neighbors_idx, 0)
                e2B, e1B_e2B_edges = self.sample_neighbor(e1B_neighbors, e1B_edges)

                # No sentences mentioning e1B and e2B that do not also mention e1A or e2A
                if len(e1B_e2B_edges) < 1:
                    continue

                mentionB, target, case = self.sample_mention(e1B_e2B_edges, 0, 2)

        return {
            'mentionA': mentionA,
            'mentionB': mentionB,
            'e1A': e1A,
            'e2A': e2A,
            'e1B': e1B,
            'e2B': e2B,
            'target': target,
            'ntokens': len(mentionA),
            'nsentences': 1,
            'ntokens_AB': len(mentionA) + len(mentionB),
            'mentionB_len': len(mentionB)
        }

    def collater(self, instances):
        if len(instances) == 0:
            return None

        mentionA_list = []
        mentionB_dict = {}
        e1A_list = []
        e2A_list = []
        e1B_list = []
        e2B_list = []
        target_list = []
        ntokens = 0
        nsentences = 0
        ntokens_AB = 0

        mentionB_len_list = [instance['mentionB_len'] for instance in instances]
        mentionB_mean = np.mean(mentionB_len_list)
        mentionB_std = np.std(mentionB_len_list)
        mentionB_max = np.max(mentionB_len_list)

        cluster_candidates = [
                                np.where(mentionB_len_list < mentionB_mean - 1.5*mentionB_std)[0], # len < mean-1.5*std
                                np.where(np.logical_and(mentionB_len_list >= mentionB_mean - 1.5*mentionB_std, mentionB_len_list < mentionB_mean - mentionB_std))[0], # mean-1.5*std <= len < mean-std
                                np.where(np.logical_and(mentionB_len_list >= mentionB_mean - mentionB_std, mentionB_len_list < mentionB_mean - 0.5*mentionB_std))[0], # mean-std <= len < mean-0.5*std
                                np.where(np.logical_and(mentionB_len_list >= mentionB_mean - 0.5*mentionB_std, mentionB_len_list < mentionB_mean))[0], # mean-0.5*std <= len < mean
                                np.where(np.logical_and(mentionB_len_list >= mentionB_mean, mentionB_len_list < mentionB_mean + 0.5*mentionB_std))[0], # mean <= len < mean+0.5*std
                                np.where(np.logical_and(mentionB_len_list >= mentionB_mean + 0.5*mentionB_std, mentionB_len_list < mentionB_mean + mentionB_std))[0], # mean+0.5*std <= len < mean+std
                                np.where(np.logical_and(mentionB_len_list >= mentionB_mean + mentionB_std, mentionB_len_list < mentionB_mean + 1.5*mentionB_std))[0], # mean+std <= len < mean+1.5*std
                                np.where(mentionB_len_list >= mentionB_mean + 1.5*mentionB_std)[0] # mean+1.5*std <= len
                             ]

        mentionB_clusters = {}
        cluster_id = 0
        for c in cluster_candidates:
            if len(c) > 0:
                mentionB_clusters[cluster_id] = c
                mentionB_dict[cluster_id] = []
                cluster_id += 1

        for i, instance in enumerate(instances):
            mentionA_list.append(instance['mentionA'])
            for cluster_id, cluster_instance_ids in mentionB_clusters.items():
                if i in cluster_instance_ids:
                    mentionB_dict[cluster_id].append(instance['mentionB'])
                    break

            e1A_list.append(instance['e1A'])
            e2A_list.append(instance['e2A'])
            e1B_list.append(instance['e1B'])
            e2B_list.append(instance['e2B'])
            target_list.append(instance['target'])
            ntokens += instance['ntokens']
            nsentences += instance['nsentences']
            ntokens_AB += instance['ntokens_AB']

        padded_mentionA = pad_sequence(mentionA_list, batch_first=True, padding_value=self.dictionary.pad())
        padded_mentionB = {}
        padded_mentionB_size = 0
        for cluster_id, cluster_mentions in mentionB_dict.items():
            padded_mentionB[cluster_id] = pad_sequence(cluster_mentions, batch_first=True, padding_value=self.dictionary.pad())
            padded_mentionB_size += torch.numel(padded_mentionB[cluster_id])

        return {
            'mentionA': padded_mentionA,
            'mentionB': padded_mentionB,
            'mentionB_size': padded_mentionB_size,
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
