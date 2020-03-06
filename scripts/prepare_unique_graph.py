from pudb import set_trace
import os
from collections import defaultdict
from itertools import combinations
import argparse
from tqdm import tqdm

import numpy as np
import torch

from fairseq.data import Dictionary, indexed_dataset
from fairseq.data.data_utils import load_indexed_dataset

def main(args):

    entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
    n_entities = len(Dictionary.load(entity_dict_path))

    print('{} entities'.format(n_entities))

    annotation_path = os.path.join(args.data_path, 'train.annotations')

    annotation_data =  load_indexed_dataset(
        annotation_path,
        None,
        dataset_impl='mmap',
    )

    print('{} sentences'.format(len(annotation_data)))

    print('starting graph construction')
    entity_neighbors, entity_edges = create_graph(annotation_data, n_entities)
    print('finished graph construction')

    neighbor_path = os.path.join(args.data_path, 'unique_neighbors')
    neighbor_len_path = os.path.join(args.data_path, 'unique_neighbors_len')
    edge_path = os.path.join(args.data_path, 'unique_edges')
    edge_len_path = os.path.join(args.data_path, 'unique_edges_len')

    neighbor_builder = indexed_dataset.make_builder(
        neighbor_path + '.bin',
        impl='mmap',
        vocab_size=n_entities
    )

    edge_builder = indexed_dataset.make_builder(
        edge_path + '.bin',
        impl='mmap',
        vocab_size=len(annotation_data)
    )

    print('creating indexed datasets')
    unique_neighbors_len = []
    unique_edges_len = []
    for entity in tqdm(range(n_entities)):
        unique_neighbors = np.unique(np.array(entity_neighbors[entity]))
        unique_neighbors_len.append(len(unique_neighbors))
        unique_edges = np.unique(np.array(entity_edges[entity]))
        unique_edges_len.append(len(unique_edges))

        neighbor_builder.add_item(torch.IntTensor(unique_neighbors))
        edge_builder.add_item(torch.IntTensor(unique_edges))

    neighbor_builder.finalize(neighbor_path + '.idx')
    edge_builder.finalize(edge_path + '.idx')

    np.save(neighbor_len_path + '.npy', unique_neighbors_len)
    np.save(edge_len_path + '.npy', unique_edges_len)

    print('finished creating indexed datasets')


def create_graph(annotation_data, n_entities):

    entity_neighbors = [list() for entity in range(n_entities)]
    entity_edges = [list() for entity in range(n_entities)]

    for sentence_idx in tqdm(range(len(annotation_data))):
        entity_ids = annotation_data[sentence_idx].reshape(-1, 3)[:, -1].numpy()

        for a, b in combinations(entity_ids, 2):
            entity_neighbors[a].append(b)
            entity_neighbors[b].append(a)

            entity_edges[a].append(sentence_idx)
            entity_edges[b].append(sentence_idx)

    return entity_neighbors, entity_edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construction of IndexedDatasets for unique graph neighbors and edges')
    parser.add_argument('--data-path', type=str, help='Data directory', default='../data/nki/bin-v3-threshold20')

    args = parser.parse_args()
    main(args)
