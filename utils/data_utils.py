import os

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