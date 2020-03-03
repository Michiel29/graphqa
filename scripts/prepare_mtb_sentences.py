import os
from collections import defaultdict
from itertools import permutations
from tqdm import tqdm
import argparse

import numpy as np
import torch

from fairseq.data import Dictionary
from fairseq.data.data_utils import load_indexed_dataset

def main(args):
    entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
    n_entities = len(Dictionary.load(entity_dict_path))

    for split in ['train', 'valid']:
        print('------------- {} -------------'.format(split))
        annotation_path = os.path.join(args.data_path, split+'.annotations')
        annotation_data =  load_indexed_dataset(
            annotation_path,
            None,
            dataset_impl='mmap',
        )
        print('{} entities\n'.format(n_entities))
        print('{} sentences\n'.format(len(annotation_data)))
        
        mtb_triplets = get_mtb_triplets(annotation_data)
        mtb_path = os.path.join(args.data_path, 'mtb_triplets_'+split)
        print('\n{} mtb triplets'.format(len(mtb_triplets)))

        print('\nsaving mtb triplets list to file\n')
        np.save(mtb_path + '.npy', mtb_triplets)

def get_mtb_triplets(annotation_data):
    print('building count dicts')
    ent_pair_counts = defaultdict(int) # for each entity pair, counts number of sentences containing it
    ent_counts = defaultdict(int) # for each entity, counts number of sentences using the entity as e1
    for sentence_id in tqdm(range(len(annotation_data))):
        entity_ids = set(annotation_data[sentence_id][2::3].numpy())
        for p in permutations(entity_ids, 2):
            ent_pair_counts[frozenset(p)] += 1
        for e in entity_ids:
            ent_counts[e] += 1

    print('\nbuilding mtb triplet list')
    mtb_triplets = []
    for sentence_id in tqdm(range(len(annotation_data))):
        entity_ids = set(annotation_data[sentence_id][2::3].numpy())
        for p in permutations(entity_ids, 2):
            case0 = ent_pair_counts[frozenset(p)] > 1 
            case1 = ent_counts[p[0]] > 1
            if case0 and case1:
                mtb_triplets.append((sentence_id, p[0], p[1]))

    return mtb_triplets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construction of IndexedDatasets for graph neighbors and edges')
    parser.add_argument('--data-path', type=str, help='Data directory', default='../data/bin_sample')

    args = parser.parse_args()
    main(args)
