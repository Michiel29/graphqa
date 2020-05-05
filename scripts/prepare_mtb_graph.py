import os
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
import torch

from fairseq.data import Dictionary, indexed_dataset
from fairseq.data.data_utils import load_indexed_dataset

import sys; sys.path.append(os.path.join(sys.path[0], '..'))
from utils.data_utils import safe_load_indexed_dataset

TAIL_ENTITY = 0
HEAD_ENTITY = 1
START_BLOCK = 6
END_BLOCK = 7
EDGE_SIZE = 8
SAMPLE_SIZE = 20

def main(args):
    entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
    n_entities = len(Dictionary.load(entity_dict_path))
    splits = ['train', 'valid']

    # Load train and valid graphs
    # edge = (right_entity, left_entity, left_start_pos, left_end_pos, right_start_pos, right_end_pos, start_block, end_block)
    print('\nLoading train and valid graphs')
    graph_dict = {}
    for split in splits:
        graph_path = os.path.join(args.data_path, split+'.graph')
        graph_dict[split] = safe_load_indexed_dataset(graph_path)

    # Load train annotation data
    # annotation = (global starting position, global ending position, sentence idx, document idx, entity idx)
    print('Loading train annotation data\n')
    annotation_path = os.path.join(args.data_path, 'train.annotations.npy')
    annotation_data = np.load(annotation_path)

    # Get number of edges per entity, for train graph
    edge_counts = {}
    for entity in tqdm(range(n_entities), desc='Getting number of edges per entity, for train graph'):
        edge_counts[entity] = int(len(graph_dict['train'][entity]) / EDGE_SIZE)

    # Get each entity's neighbors, for train graph
    neighbors = []
    for entity in tqdm(range(n_entities), desc="Getting each entity's neighbors, for train graph"):
        if edge_counts[entity] == 0:
            neighbors.append(set())
            continue
        cur_neighbors = graph_dict['train'][entity][TAIL_ENTITY::EDGE_SIZE].numpy()
        neighbors.append(set(cur_neighbors))

    # Get all edge start/end indices, for train graph
    all_start_blocks, all_end_blocks = [], []
    for entity in tqdm(range(n_entities), desc='Getting edge start/end indices, for train graph'):
        cur_start_blocks = graph_dict['train'][entity][START_BLOCK::EDGE_SIZE].numpy()
        cur_end_blocks = graph_dict['train'][entity][END_BLOCK::EDGE_SIZE].numpy()
        all_start_blocks += list(cur_start_blocks)
        all_end_blocks += list(cur_end_blocks)
        assert len(cur_start_blocks) == edge_counts[entity]
    
    # Get all edge start/end annotation indices, for train graph (vectorize np.searchsorted for faster runtime)
    timer_start = timer()
    all_s, all_e = get_annotation_block_indices(annotation_data, all_start_blocks, all_end_blocks)
    timer_end = timer()
    print('Finished get_annotation_block_indices in ' + str(timedelta(seconds=timer_end-timer_start)) + '\n')

    # Build MTB graph
    for split in splits:
        build_mtb_graph(
            graph_dict[split],
            graph_dict['train'],
            annotation_data,
            all_s, 
            all_e,
            edge_counts, 
            neighbors,
            args.data_path,
            split
        )

def get_annotation_block_indices(annotation_data, start_block, end_block):
    # From http://sociograph.blogspot.com/2011/12/gotcha-with-numpys-searchsorted.html
    start_block = annotation_data.dtype.type(start_block)
    end_block = annotation_data.dtype.type(end_block)

    # We are interested in all annotations that INTERSECT [start_block; end_block)
    # Recall that the [start_pos; end_pos) interval for the annotation s is defined as
    # [annotations[s - 1][0], annotations[s - 1][1])
    #
    # From https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html
    # side	returned index i satisfies
    # left	a[i-1] < v <= a[i]
    # right	a[i-1] <= v < a[i]
    #
    # First, we need to find an index s such that
    # annotations[s - 1].end_pos <= start_block < annotations[s].end_pos
    s = np.searchsorted(annotation_data[:, 1], start_block, side='right')

    # Second, we need to find an index e such that
    # annotations[e - 1].start_pos < end_block <= annotations[e].start_pos
    e = np.searchsorted(annotation_data[:, 0], end_block, side='left')

    return s, e

def build_mtb_graph(graph_split, graph_train, annotation_data_train, all_s_train, all_e_train, edge_counts_train, neighbors_train, data_path, split):
    n_entities = len(edge_counts_train)
    min_pos = 2 if split == 'train' else 1 # train graph needs to contain at least min_pos positive texts, besides the base text
    min_strong_neg = 1 # train graph needs to contain at least min_strong_neg strong negative texts

    # Initialize MTB indexed dataset
    mtb_graph_path = os.path.join(data_path, 'mtb_' + split + '.graph')
    mtb_graph_builder = indexed_dataset.MMapIndexedDatasetBuilder(
        mtb_graph_path + '.bin',
        dtype=np.int64,
    )

    # Initialize global edge start/end indices to zero
    edge_count_train_start, edge_count_train_end = 0, 0

    # Iterate through all head entities
    for head in tqdm(range(n_entities), desc='Building {} MTB graph'.format(split)):
        
        # Update edge_count start/end indices
        edge_count_train_start = edge_count_train_end
        edge_count_train_end += edge_counts_train[head]
           
        # If we know current head entity cannot have positives or strong negatives, add empty list to MTB indexed dataset
        if len(graph_split[head]) == 0 or edge_counts_train[head] < min_pos + min_strong_neg:
            mtb_graph_builder.add_item(torch.LongTensor([]))
            continue

        # Get start/end indices for all edges of current head entity
        cur_s_train = all_s_train[edge_count_train_start:edge_count_train_end]
        cur_e_train = all_e_train[edge_count_train_start:edge_count_train_end]

        # Get edges for current head entity, for the graph of current split
        cur_edges_split = graph_split[head].numpy().reshape(-1, EDGE_SIZE)
        cur_edges_train = graph_train[head].numpy().reshape(-1, EDGE_SIZE)

        # For each of head's tail candidates, count how many edges contain the tail
        edge_tails_counts_train = defaultdict(int)
        cur_tails_train = cur_edges_train[:, TAIL_ENTITY]
        edge_tails_counts_train.update(dict(Counter(cur_tails_train)))

        # Initialize MTB edges list for current head entity
        cur_mtb_edges = []

        # Iterate through all edges of the current head entity
        for edge_idx in range(cur_edges_split.shape[0]):

            # Get current edge
            cur_edge_split = cur_edges_split[edge_idx]

            # Get current tail entity
            tail = cur_edge_split[TAIL_ENTITY]

            # Check for positives (i.e., are there at least two texts mentioning both head and tail?)
            pos = edge_tails_counts_train[tail] >= min_pos
            if pos is False:
                continue

            # Check for strong negatives: 
            # - First, does head have at least one neighbor which is not also neighbors with tail?
            # - If not, does head have at least one edge which does not contain tail? 
            #   (We approximate this by checking a sample of SAMPLE_SIZE of head's edges.)
            head_neighbors, tail_neighbors = neighbors_train[head], neighbors_train[tail]
            
            # Temporarily remove tail and head from head_neighbors and tail_neighbors, respectively
            head_neighbors.discard(tail) 
            tail_neighbors.discard(head)

            # Check if all neighbors of head are also neighbors of tail
            if not head_neighbors.issubset(tail_neighbors):
                strong_neg = True
            else:
                strong_neg = False

            # Add tail and head back to head_neighbors and tail_neighbors, respectively
            head_neighbors.add(tail)
            tail_neighbors.add(head)

            # If head_neighbors is not a subset of tail_neighbors, check if head has at least one edge which does not contain tail
            if not strong_neg:
                sample = np.random.choice(len(cur_s_train), size=min(SAMPLE_SIZE, len(cur_s_train)), replace=False)
                for edge_idx_ in sample:
                    cur_edge_train_entities = set(annotation_data_train[slice(cur_s_train[edge_idx_], cur_e_train[edge_idx_])][:, -1])
                    assert head in cur_edge_train_entities
                    assert len(cur_edge_train_entities) > 1
                    if tail not in cur_edge_train_entities:
                        strong_neg = True
                        break
                    
            # If pos and strong_neg are both true, then add the current edge to cur_mtb_edges
            if pos and strong_neg:
                cur_mtb_edges.append(tuple(cur_edge_split))
        
        # Add current entity's sorted MTB edges to the MTB graph
        cur_mtb_edges.sort()
        mtb_graph_builder.add_item(torch.LongTensor(cur_mtb_edges))

    mtb_graph_builder.finalize(mtb_graph_path + '.idx')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Constructing arrays of MTB triplets satisfying pos and strong_neg')
    parser.add_argument('--data-path', type=str, help='Data directory', default='../data/nki/bin-v5-threshold20')

    args = parser.parse_args()
    main(args)
