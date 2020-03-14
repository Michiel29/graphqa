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

import sys; sys.path.append(os.path.join(sys.path[0], '..'))
from utils.data_utils import safe_load_indexed_dataset

def main(args):
    entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
    n_entities = len(Dictionary.load(entity_dict_path))
    splits = ['train', 'valid']

    # Load unique neighbors
    neighbors_path = os.path.join(args.data_path, 'neighbors')
    neighbors = safe_load_indexed_dataset(neighbors_path)
    neighbors = [set(n.numpy()) for n in neighbors]

    # Load unique edges
    edges_path = os.path.join(args.data_path, 'edges')
    edges = safe_load_indexed_dataset(edges_path)
    edges = [np.unique(e.numpy()) for e in edges]

    # Load annotation data
    annotation_dict = {}
    for split in splits:
        annotation_path = os.path.join(args.data_path, split+'.annotations')
        annotation_data = safe_load_indexed_dataset(annotation_path)
        print('{} {} texts'.format(len(annotation_data), split))
        annotation_dict[split] = annotation_data

    # Build unique entities dict
    entities_dict = {}
    print('\nbuilding unique entities dict')
    for split in splits:
        print(split)
        entities_dict[split] = get_unique_entities(annotation_dict[split])

    # Build entity pair count dict
    print('\nbuilding entity pair count dict')
    ent_pair_counts = get_entity_pair_counts(len(annotation_dict['train']), entities_dict['train'])

    # Build MTB triplets list
    for split in splits:
        print('\n------------- {} -------------'.format(split))
        print('{} entities'.format(n_entities))

        print('building mtb triplets list')
        mtb_triplets = get_mtb_triplets(ent_pair_counts, 
                                        len(annotation_dict[split]), 
                                        entities_dict[split], 
                                        entities_dict['train'], 
                                        neighbors, edges)

        print('saving mtb triplets list to file')
        mtb_path = os.path.join(args.data_path, 'mtb_triplets_'+split)
        print('{} mtb triplets'.format(len(mtb_triplets)))
        np.save(mtb_path + '.npy', mtb_triplets)

def get_unique_entities(annotation_data):
    unique_entities = [list() for text in range(len(annotation_data))]
    for text_idx in tqdm(range(len(annotation_data))):
        entity_ids = list(set(annotation_data[text_idx][2::3].numpy()))
        unique_entities[text_idx] += entity_ids

    return unique_entities

def get_entity_pair_counts(n_train_texts, entities):
    ent_pair_counts = defaultdict(int) # for each directed entity pair, counts the number of texts containing it
    for text_id in tqdm(range(n_train_texts)):
        for p in permutations(entities[text_id], 2):
            ent_pair_counts[p] += 1

    return ent_pair_counts

def get_mtb_triplets(ent_pair_counts, n_texts, entities_split, entities_train, neighbors, edges):
    mtb_triplets = []
    for text_id in tqdm(range(n_texts)):
        for p in permutations(entities_split[text_id], 2):
            # Check for positives (i.e., are there at least two texts mentioning both head and tail?)
            pos = ent_pair_counts[p] > 1 
            if pos is False:
                continue
            
            # Check for strong negatives: 
            # - First, does head have at least one neighbor which is not also neighbors with tail?
            # - If not, does head have at least one edge which does not contain tail? 
            #   (We approximate this by checking a sample of 20 of head's edges.)
            head, tail = p[0], p[1]
            head_neighbors, tail_neighbors = neighbors[head], neighbors[tail]
            
            # Temporarily remove tail and head from head_neighbors and tail_neighbors, respectively
            head_neighbors.discard(tail) 
            tail_neighbors.discard(head)

            # Check if all neighbors of head are also neighbors of tail
            if not head_neighbors.issubset(tail_neighbors):
                strong_neg = True
            else:
                strong_neg = False

            # If head_neighbors is not a subset of tail_neighbors, check if head has at least one edge which does not contain tail
            if not strong_neg:
                head_edges = edges[head]
                head_edges_sample = np.random.choice(len(head_edges), size=min(20, len(head_edges)), replace=False)
                for idx in head_edges_sample:
                    cur_entities = entities_train[head_edges[idx]]
                    if tail not in cur_entities:
                        strong_neg = True
                        break
            
            # Add tail and head back to head_neighbors and tail_neighbors, respectively
            head_neighbors.add(tail)
            tail_neighbors.add(head)
                    
            # If pos and strong_neg are both true, then add the current triplet to mtb_triplets
            if pos and strong_neg:
                mtb_triplets.append((text_id, head, tail))

    return mtb_triplets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Constructing arrays of MTB triplets satisfying pos and strong_neg')
    parser.add_argument('--data-path', type=str, help='Data directory', default='../data/nki/bin-v3-threshold20')

    args = parser.parse_args()
    main(args)
