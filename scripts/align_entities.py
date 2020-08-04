# This file loads base and external entity dictionaries and array of values indexed by external id. Maps each entity in external to corresponding entity in base. Finally, creates an array of values indexed by base entity id.
import os
import argparse

import numpy as np
from tqdm import tqdm

from fairseq.data import Dictionary

class EntityDictionary(Dictionary):
    """Dictionary with no special tokens"""
    def __init__(self):
        self.symbols = []
        self.count = []
        self.indices = {}
        self.unk_index = None

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:
        <symbol0> <count0>
        <symbol1> <count1>
        """
        d = cls()
        d.add_from_file(f)
        return d

def main(args):
    # Load base and external entity dictionaries.
    base_dictionary = EntityDictionary.load(os.path.join(args.base_path, 'entity.dict.txt'))
    base_to_external_map = np.zeros(len(base_dictionary), dtype=np.int64) - 1


    with open(os.path.join(args.external_path, 'dict.entity')) as f:
        lines = f.readlines()
        external_to_base_map = np.zeros(len(lines), dtype=np.int64) - 1
        for idx, line in enumerate(lines):
            entity_name = line.split('\t')[0][22:]
            base_idx = base_dictionary.index(entity_name)
            if base_idx is not None:
                base_to_external_map[base_idx] = idx
                external_to_base_map[idx] = base_idx

    external_candidate_path = os.path.join(args.external_path, 'entity.candidates.idx.npy')
    external_candidates = np.load(external_candidate_path)
    external_score_path = os.path.join(args.external_path, 'entity.scores.idx.npy')
    external_scores = np.load(external_score_path)

    external_shape = external_candidates.shape
    new_shape = (len(base_dictionary),) + external_shape[1:]
    new_candidates = np.zeros(new_shape, dtype=external_candidates.dtype) - 1
    new_scores = np.zeros(new_shape, dtype=external_scores.dtype) - 1

    for base_idx in tqdm(range(len(base_to_external_map)), desc='remapping entities'):
        external_entity = base_to_external_map[base_idx]
        if not external_entity == -1:
            new_candidates[base_idx] = external_to_base_map[external_candidates[external_entity]]
            new_scores[base_idx] = external_scores[external_entity]

    print('saving remapped entity files')
    new_external_candidate_path = external_candidate_path[:-8] + '_remap.idx'
    np.save(new_external_candidate_path, new_candidates)
    new_external_score_path = external_score_path[:-8] + '_remap.idx'
    np.save(new_external_score_path, new_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=str, help='Data directory', default='../data/nki/bin-v5-threshold20')
    parser.add_argument('--external-path', type=str, default='../data/lama')

    args = parser.parse_args()
    main(args)
