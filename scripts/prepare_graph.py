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
    graph_path = os.path.join(args.data_path, 'graph')
    entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
    n_entities = len(Dictionary.load(entity_dict_path))

    print('{} entities'.format(n_entities))

    annotation_path = os.path.join(args.data_path, 'train.annotations')

    annotation_data =  load_indexed_dataset(
        annotation_path,
        None,
        dataset_impl='mmap',
    )

    print('{} sentences\n'.format(len(annotation_data)))

    print('starting graph construction')
    entity_neighbors, entity_edges = create_graph(annotation_data, n_entities)
    (
        index_to_entity_pair,
        index_text_count,
        index_to_sentences,
        max_sentence_id,
     ) = count_edges(entity_neighbors, entity_edges)
    print('finished graph construction\n')

    os.makedirs(graph_path, exist_ok=True)
    neighbor_path = os.path.join(graph_path, 'neighbors')
    edge_path = os.path.join(graph_path, 'edges')
    index_to_entity_pair_path = os.path.join(graph_path, 'index_to_entity_pair')
    index_text_count_path = os.path.join(graph_path, 'index_text_count')
    index_to_sentences_path = os.path.join(graph_path, 'index_to_sentences')

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

    print('creating indexed datasets for neighbors/edges')
    for entity in tqdm(range(n_entities)):

        neighbors = np.array(entity_neighbors[entity])
        sorted_indices = np.argsort(neighbors)

        sorted_neighbors = neighbors[sorted_indices]
        sorted_edges = np.array(entity_edges[entity])[sorted_indices]

        neighbor_builder.add_item(torch.IntTensor(sorted_neighbors))
        edge_builder.add_item(torch.IntTensor(sorted_edges))

    neighbor_builder.finalize(neighbor_path + '.idx')
    edge_builder.finalize(edge_path + '.idx')

    print('creating indexed datasets for index_to_entity_pair/index_text_count')
    np.save(index_to_entity_pair_path, index_to_entity_pair)
    np.save(index_text_count_path, index_text_count)

    print('creating indexed datasets for index_to_sentences')
    index_to_sentences_builder = indexed_dataset.make_builder(
        index_to_sentences_path + '.bin',
        impl='mmap',
        vocab_size=max_sentence_id,
    )
    for edge_index in tqdm(range(len(index_to_sentences))):
        index_to_sentences_builder.add_item(
            torch.IntTensor(index_to_sentences[edge_index])
        )
    index_to_sentences_builder.finalize(index_to_sentences_path + '.idx')

    print('finished creating indexed datasets')


def create_graph(annotation_data, n_entities):

    entity_neighbors = [list() for entity in range(n_entities)]
    entity_edges = [list() for entity in range(n_entities)]

    for sentence_idx in tqdm(range(len(annotation_data))):
        entity_ids = set(annotation_data[sentence_idx].reshape(-1, 3)[:, -1].numpy())

        for a, b in combinations(entity_ids, 2):
            entity_neighbors[a].append(b)
            entity_neighbors[b].append(a)

            entity_edges[a].append(sentence_idx)
            entity_edges[b].append(sentence_idx)

    return entity_neighbors, entity_edges

def count_edges(entity_neighbors, entity_edges):
    entity_pair_to_index = {}

    print('count_edges: indexing entity pairs')
    for v, neighbors in tqdm(enumerate(entity_neighbors), total=len(entity_neighbors)):
        unique_neighbors = np.unique(neighbors)
        for u in unique_neighbors:
            edge = frozenset((v, u))
            if edge not in entity_pair_to_index:
                entity_pair_to_index[edge] = len(entity_pair_to_index)
    print('count_edges: collected %d undirected edges' % len(entity_pair_to_index))

    print('count_edges: building index -> entity pair')
    index_to_entity_pair = np.zeros((len(entity_pair_to_index), 2), dtype=np.int32)
    for edge, index in tqdm(entity_pair_to_index.items()):
        v = min(edge)
        u = max(edge)
        index_to_entity_pair[index][0] = v
        index_to_entity_pair[index][1] = u

    print('count_edges: building index -> sentence')
    max_sentence_id = -1
    index_to_sentences = [list() for edge in range(len(entity_pair_to_index))]
    for v, (neighbors, sentences) in tqdm(
        enumerate(zip(entity_neighbors, entity_edges)),
        total=len(entity_neighbors),
    ):
        for u, s in zip(neighbors, sentences):
            if v <= u:
                index_to_sentences[entity_pair_to_index[frozenset((v, u))]].append(s)
                max_sentence_id = max(max_sentence_id, s)
    print('count_edges: built index -> sentence: max sentence idx = %d' % max_sentence_id)

    print('count_edges: counting edges')
    index_text_count = np.zeros(len(entity_pair_to_index), dtype=np.int32)
    for v, edge in tqdm(enumerate(entity_neighbors), total=len(entity_neighbors)):
        for u in edge:
            if v <= u:
                index_text_count[entity_pair_to_index[frozenset((v, u))]] += 1
    return index_to_entity_pair, index_text_count, index_to_sentences, max_sentence_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construction of IndexedDatasets for graph neighbors and edges')
    parser.add_argument('--data-path', type=str, help='Data directory', default='../data/bin_sample')

    args = parser.parse_args()
    main(args)
