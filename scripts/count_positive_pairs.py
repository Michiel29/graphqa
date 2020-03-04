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
        ent_pair_counts = get_entity_pair_counts(annotation_data_dict['train'], entities['train'])
        print('saving entity pair count dict')
        with open(ent_pair_counts_path, 'wb') as f:
            pickle.dump(ent_pair_counts, f, pickle.HIGHEST_PROTOCOL)

    for split in splits:
        print('\n------------- {} -------------'.format(split))
        print('{} entities'.format(n_entities))

        print('counting positive pairs')
        n_pos_pairs, n_pos_pairs_thresh, max_n, max_ent_pair = count_positive_pairs(ent_pair_counts, annotation_data_dict[split], entities_dict[split], neighbors, neighbors_len, edges, edges_len)

        print('number of positive pairs: {}'.format(int(n_pos_pairs)))
        print('number of positive pairs (thresh): {}'.format(n_pos_pairs_thresh))
        print('max number of positive pairs per entity pair: {}'.format(int(max_n)))
        print('ent pair with largest number of positive pairs: {}'.format(max_ent_pair))

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
    ent_pair_counts = defaultdict(int) # for each entity pair, counts number of sentences containing it
    for sentence_id in tqdm(range(len(annotation_data))):
        for p in permutations(entities[sentence_id].numpy(), 2):
            ent_pair_counts[frozenset(p)] += 1

    return ent_pair_counts

def count_positive_pairs(ent_pair_counts, annotation_data, entities, neighbors, neighbors_len, edges, edges_len):
    n_pos_pairs = 0 
    n_pos_pairs_thresh = defaultdict(int)
    k_vals = [10, 25, 50, 100, 250, 500, 1000]
    max_n = 0
    for ent_pair, count in tqdm(ent_pair_counts.items()):
        cur_n = 0.5 * count * (count-1)
        n_pos_pairs += cur_n
        for k in k_vals:
            n_pos_pairs_thresh[k] += min(k, cur_n)
        if cur_n > max_n:
            max_n = cur_n
            max_ent_pair = ent_pair

    return n_pos_pairs, n_pos_pairs_thresh, max_n, max_ent_pair

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Constructing arrays of MTB triplets satisfying case0 and case1')
    parser.add_argument('--data-path', type=str, help='Data directory', default='../data/nki/bin-v3-threshold20')

    args = parser.parse_args()
    main(args)
