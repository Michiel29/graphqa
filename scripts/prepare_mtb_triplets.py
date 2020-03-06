import os
from collections import defaultdict
from itertools import permutations
from tqdm import tqdm
import argparse
import pickle

import numpy as np
import torch

from fairseq.data import Dictionary
from fairseq.data.data_utils import load_indexed_dataset


def main(args):

    entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
    n_entities = len(Dictionary.load(entity_dict_path))
    splits = ['train', 'valid']
    entities_dict, neighbors, neighbors_len, edges, edges_len = load_helper_datasets(args.data_path, splits)

    annotation_data_dict = {}
    for split in splits:
        annotation_path = os.path.join(args.data_path, split+'.annotations')
        annotation_data = load_indexed_dataset(
            annotation_path,
            None,
            dataset_impl='mmap',
        )
        print('{} {} sentences'.format(len(annotation_data), split))
        annotation_data_dict[split] = annotation_data
        
    ent_pair_counts_path = os.path.join(args.data_path, 'ent_pair_counts.pkl')
    if os.path.exists(ent_pair_counts_path):
        print('loading entity pair count dict')
        with open(ent_pair_counts_path, 'rb') as f:
            ent_pair_counts = pickle.load(f)
    else:
        print('building entity pair count dict')
        ent_pair_counts = get_entity_pair_counts(annotation_data_dict['train'], entities_dict['train'])
        print('saving entity pair count dict')
        with open(ent_pair_counts_path, 'wb') as f:
            pickle.dump(ent_pair_counts, f, pickle.HIGHEST_PROTOCOL)

    for split in splits:
        print('\n------------- {} -------------'.format(split))
        print('{} entities'.format(n_entities))

        print('building mtb triplets list')
        mtb_triplets = get_mtb_triplets(ent_pair_counts, annotation_data_dict[split], entities_dict[split], neighbors, neighbors_len, edges, edges_len)

        print('saving mtb triplets list to file')
        mtb_path = os.path.join(args.data_path, 'mtb_triplets_'+split)
        print('{} mtb triplets'.format(len(mtb_triplets)))
        np.save(mtb_path + '.npy', mtb_triplets)

def load_helper_datasets(data_path, splits):
    entities_dict = {}
    for split in splits:
        entities_path = os.path.join(data_path, '_'.join(['unique_entities', split]))
        entities = load_indexed_dataset(
            entities_path,
            None,
            dataset_impl='mmap'
        )   
        if entities is None:
            raise FileNotFoundError('Unique entites dataset not found: {}'.format(entities_path))
        entities_dict[split] = entities
    
    neighbor_path = os.path.join(data_path, 'unique_neighbors')
    neighbor_len_path = os.path.join(data_path, 'unique_neighbors_len.npy')
    edge_path = os.path.join(data_path, 'unique_edges')
    edge_len_path = os.path.join(data_path, 'unique_edges_len.npy')

    neighbors = load_indexed_dataset(
        neighbor_path,
        None,
        dataset_impl='mmap'
    )   
    if neighbors is None:
        raise FileNotFoundError('Unique neighbors dataset not found: {}'.format(neighbor_path))

    neighbors_len = np.load(neighbor_len_path)
    if neighbors_len is None:
        raise FileNotFoundError('Unique neighbor lengths dataset not found: {}'.format(neighbor_len_path))

    edges = load_indexed_dataset(
        edge_path,
        None,
        dataset_impl='mmap'
    )   
    if edges is None:
        raise FileNotFoundError('Unique edges dataset not found: {}'.format(edge_path))
    
    edges_len = np.load(edge_len_path)
    if edges_len is None:
        raise FileNotFoundError('Unique edge lengths dataset not found: {}'.format(edge_len_path))

    return entities_dict, neighbors, neighbors_len, edges, edges_len

def get_entity_pair_counts(annotation_data, entities):
    ent_pair_counts = defaultdict(int) # for each directed entity pair, counts the number of sentences containing it
    for sentence_id in tqdm(range(len(annotation_data))):
        for p in permutations(entities[sentence_id].numpy(), 2):
            ent_pair_counts[p] += 1

    return ent_pair_counts

def get_mtb_triplets(ent_pair_counts, annotation_data, entities, neighbors, neighbors_len, edges, edges_len):
    mtb_triplets = []
    for sentence_id in tqdm(range(len(annotation_data))):
        for p in permutations(entities[sentence_id].numpy(), 2):
            # Check case0 (i.e., are there at least two sentences mentioning both e1 and e2?)
            case0 = ent_pair_counts[p] > 1 
            if case0 is False:
                continue

            # Check case1 (i.e., does e1 have at least one neighbor which is not also neighbors with e2?)
            if neighbors_len[p[0]] < 2: # e1 doesn't have any neighbors besides e2 
                case1 = False
            elif neighbors_len[p[0]] > neighbors_len[p[1]]: # e1 has more neighbors than e2 does --> e1 has at least one neighbor which isn't in e2's neighbor list
                case1 = True
            else:
                e1_neighbors, e2_neighbors = neighbors[p[0]].numpy(), neighbors[p[1]].numpy()
                e1_neighbors, e2_neighbors = e1_neighbors[e1_neighbors != p[1]], e2_neighbors[e2_neighbors != p[0]]
                if e1_neighbors[0] < e2_neighbors[0]: # e1's lowest-index neighbor is not in e2's neighbor list 
                    case1 = True
                elif e1_neighbors[-1] > e2_neighbors[-1]: # e1's highest-index neighbor is not in e2's neighbor list
                    case1 = True
                else:
                    for i in range(1, neighbors_len[p[0]]-1): # iterate through all of e1's neighbors, besides the first and last ones (we already checked those)
                        if e1_neighbors[i] not in e2_neighbors[1:-1]: # e1 has a neighbor which is not in e2's neighbor list
                            case1 = True
                            break
                        elif i == neighbors_len[p[0]]-2: # all of e1's neighbors are in e2's neighbor list
                            case1 = False
            
            # If case0 and case1 are both true, then add the current triplet to mtb_triplets
            if case0 and case1:
                mtb_triplets.append((sentence_id, p[0], p[1]))

    return mtb_triplets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Constructing arrays of MTB triplets satisfying case0 and case1')
    parser.add_argument('--data-path', type=str, help='Data directory', default='../data/nki/bin-v3-threshold20')

    args = parser.parse_args()
    main(args)
