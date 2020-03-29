import os
import numpy as np
from contextlib import contextmanager

from fairseq.data import data_utils, PrependTokenDataset


def safe_load_indexed_dataset(path):
    data =  data_utils.load_indexed_dataset(
        path,
        None,
        dataset_impl='mmap',
    )
    if data is None:
        raise FileNotFoundError('Dataset not found: {}'.format(path))
    return data


def load_annotated_text(data_path, prefix, bos):
    return (
        PrependTokenDataset(
            safe_load_indexed_dataset(
                os.path.join(data_path, prefix + '.text'),
            ),
            bos,
        ),
        safe_load_indexed_dataset(
            os.path.join(data_path, prefix + '.annotations'),
        ),
    )

@contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return

    def make_hashable(hash_input): # generates integer representation from string for seed
        if isinstance(hash_input, str):
            hash_input = int(''.join([str(ord(char)) for char in hash_input]))
        return hash_input

    seed_list = [make_hashable(seed)] + [make_hashable(add_seed) for add_seed in addl_seeds]
    seed = int(hash(tuple(seed_list)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

# From https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_arrays(arrays, seed=None):
    """Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    seed : Seed value if it is None, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if seed is None else seed

    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)