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

    for split in ['train', 'valid']:
        print('\n------------- {} -------------'.format(split))
        annotation_path = os.path.join(args.data_path, split+'.annotations')
        annotation_data =  load_indexed_dataset(
            annotation_path,
            None,
            dataset_impl='mmap',
        )

        print('{} sentences'.format(len(annotation_data)))

        print('starting to get unique entities')
        unique_entities = get_unique_entities(annotation_data)
        print('finished getting unique entities')

        entities_path = os.path.join(args.data_path, '_'.join(['unique_entities', split]))

        entities_builder = indexed_dataset.make_builder(
            entities_path + '.bin',
            impl='mmap',
            vocab_size=len(annotation_data)
        )

        print('creating indexed datasets')
        for sentence_idx in tqdm(range(len(annotation_data))):
            entities_builder.add_item(torch.IntTensor(unique_entities[sentence_idx]))

        entities_builder.finalize(entities_path + '.idx')
        print('finished creating indexed datasets')


def get_unique_entities(annotation_data):

    unique_entities = [list() for sentence in range(len(annotation_data))]

    for sentence_idx in tqdm(range(len(annotation_data))):
        entity_ids = list(set(annotation_data[sentence_idx][2::3].numpy()))

        unique_entities[sentence_idx] += entity_ids

    return unique_entities


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Construction of IndexedDataset for each sentence's unique entities")
    parser.add_argument('--data-path', type=str, help='Data directory', default='../data/nki/bin-v3-threshold20')

    args = parser.parse_args()
    main(args)
