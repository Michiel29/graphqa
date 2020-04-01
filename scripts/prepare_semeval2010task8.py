import argparse
import json
import codecs
import copy
import numpy as np
import os
import tqdm
from operator import itemgetter

import torch

from fairseq.data import Dictionary
from fairseq.data import indexed_dataset
from fairseq.data.encoders.gpt2_bpe import get_encoder

np.random.seed(0)

TRAINING_TQDM_BAD_FORMAT = (
    '{l_bar}{bar}| '
    '{n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]'
)


class SemEval2010Task8Processor(object):
    def __init__(self, roberta_dir, dataset_impl, append_eos):
        self.dataset_impl = dataset_impl
        self.append_eos = append_eos
        self.roberta_dir = roberta_dir
        assert os.path.isdir(self.roberta_dir)

        self.ent_tokens = {
            1: {'start': '<e1>', 'end': '</e1>'},
            2: {'start': '<e2>', 'end': '</e2>'},
        }

    def initializer(self):
        global bpe
        bpe = get_encoder(
            os.path.join(self.roberta_dir, 'gpt2_bpe', 'encoder.json'),
            os.path.join(self.roberta_dir, 'gpt2_bpe', 'vocab.bpe'),
        )
        global vocab
        vocab = Dictionary.load(os.path.join(self.roberta_dir, 'roberta.base', 'dict.txt'))

    def generate_tmp_filename(self):
        tf = tempfile.NamedTemporaryFile(
            dir=self.tmp,
            delete=True,
        )
        tf.close()
        return tf.name

    def assert_text_and_surface_form_are_equal(self, words, surface_form):
        if isinstance(words, list):
            text = ' '.join(words)
        else:
            assert isinstance(words, str)
            text = words
        if text.startswith(' '):
            text = text[1:]
        text = text.lower()

        assert text == surface_form

    def strip_empty_words(self, words, annotations):
        new_words = []
        new_annotations = copy.deepcopy(annotations)

        for word_id, word in enumerate(words):
            if len(word.strip()) == 0:
                for i in range(len(annotations)):
                    if annotations[i]['start_word'] >= word_id:
                        new_annotations[i]['start_word'] -= 1
                    if annotations[i]['end_word'] > word_id:
                        new_annotations[i]['end_word'] -= 1
            else:
                new_words.append(word)
        for annotation in new_annotations:
            self.assert_text_and_surface_form_are_equal(
                new_words[annotation['start_word']:annotation['end_word']],
                annotation['surface_form'],
            )

        return new_words, new_annotations

    def apply_gt2_bpe(self, sentence, annotations):
        global bpe
        ids = list(map(str, bpe.encode(sentence)))
        word_index = 0
        current_annotation_index, next_annotation_index = None, 0

        for token_id, token in enumerate(ids):
            if bpe.decode([int(token)]).startswith(' '):
                word_index += 1

            if current_annotation_index is not None and annotations[current_annotation_index]['end_word'] == word_index:
                annotations[current_annotation_index]['end_word'] = token_id
                current_annotation_index = None
                next_annotation_index += 1
                if next_annotation_index >= len(annotations):
                    break

            if current_annotation_index is None and annotations[next_annotation_index]['start_word'] == word_index:
                annotations[next_annotation_index]['start_word'] = token_id
                current_annotation_index = next_annotation_index

        if current_annotation_index is not None:
            annotations[current_annotation_index]['end_word'] = len(ids)

        for annotation in annotations:
            decoded = bpe.decode([int(x) for x in ids[annotation['start_word']:annotation['end_word']]])
            self.assert_text_and_surface_form_are_equal(
                decoded,
                annotation['surface_form'],
            )

        return ids, annotations

    def filter_tokens_and_build_annotations(self, tokens):
        start_idx, end_idx = {}, {}

        for ent_id in [1, 2]:
            start_idx_tmp = tokens.index(self.ent_tokens[ent_id]['start'])
            end_idx_tmp = tokens.index(self.ent_tokens[ent_id]['end'])
            if end_idx_tmp - start_idx_tmp == 1:
                tokens.insert(end_idx_tmp, '<blank>')

        for ent_id in [1, 2]:
            start_idx[ent_id] = tokens.index(self.ent_tokens[ent_id]['start'])
            end_idx[ent_id] = tokens.index(self.ent_tokens[ent_id]['end'])

        if start_idx[1] < start_idx[2]:
            end_idx[1] -= 1
            start_idx[2] -= 2
            end_idx[2] -= 3
        else:
            start_idx[1] -= 2
            end_idx[1] -= 3
            end_idx[2] -= 1
        
        new_tokens = copy.deepcopy(tokens)
        new_tokens = [x for x in new_tokens if x not in list(self.ent_tokens[1].values()) + list(self.ent_tokens[2].values())]
        annotations = []
        for ent_id in [1, 2]:
            annotations.append({   
                'surface_form': ' '.join(new_tokens[start_idx[ent_id]:end_idx[ent_id]]).lower(),
                'start_word': start_idx[ent_id],
                'end_word': end_idx[ent_id],
                'uri': ent_id
            })

        return new_tokens, annotations

    def __call__(self, text):
        global vocab
        text = text.replace('<e1>', ' <e1> ').replace('</e1>', ' </e1> ').replace('<e2>', ' <e2> ').replace('</e2>', ' </e2> ')
        tokens = text.split()
        tokens, annotations = self.filter_tokens_and_build_annotations(tokens)
        annotations.sort(key=lambda x: x['start_word'])
        words, annotations = self.strip_empty_words(tokens, annotations)
        ids, annotations = self.apply_gt2_bpe(' '.join(words), annotations)
        ids_tensor = vocab.encode_line(line=' '.join(ids), append_eos=self.append_eos)

        assert len(ids_tensor) == len(ids) + int(self.append_eos)
        annotations_list = [
            [annotation['start_word'], annotation['end_word'], 0, 0, annotation['uri']]
            for annotation in annotations
        ]
        return ids_tensor, np.array(annotations_list, dtype=np.int64)


def main(args):
    # if args.split in ['train', 'dev']:
    #     data_path = os.path.join(args.root_dir, 'SemEval2010_task8_training', 'TRAIN_FILE.TXT')
    #     rand_indices = np.random.permutation(8000)
    #     train_indices, dev_indices = rand_indices[:6500], rand_indices[6500:]
    # elif args.split == 'test':
    #     data_path = os.path.join(args.root_dir, 'SemEval2010_task8_testing_keys', 'TEST_FILE_FULL.TXT')

    if args.split == 'train':
        data_path = os.path.join(args.root_dir, 'SemEval2010_task8_training', 'TRAIN_FILE.TXT')
    elif args.split == 'dev':
        data_path = os.path.join(args.root_dir, 'SemEval2010_task8_testing_keys', 'TEST_FILE_FULL.TXT')
    else:
        raise NotImplementedError

    with open(data_path, 'r') as f:
        data = f.read().splitlines()
        
    data = [x for x in data if x != '']
    texts = [x.split('\t')[1][1:-1] for x in data[0::3]]
    relation_types = [x.replace(' ', '') for x in data[1::3]]
    unique_relation_types = sorted(list(set(relation_types)))
    unique_relation_types.remove('Other')
    unique_relation_types.append('Other')

    # if args.split == 'train':
    #     texts = list(itemgetter(*train_indices)(texts))
    #     relation_types = list(itemgetter(*train_indices)(relation_types))
    # elif args.split == 'dev':
    #     texts = list(itemgetter(*dev_indices)(texts))
    #     relation_types = list(itemgetter(*dev_indices)(relation_types))

    print('-- Loaded data from %s' % data_path)

    processor = SemEval2010Task8Processor(
        args.roberta_dir,
        args.dataset_impl,
        args.append_eos,
    )
    pbar = tqdm.tqdm(
        total=len(texts),
        desc='Processing Wiki',
        bar_format=TRAINING_TQDM_BAD_FORMAT,
    )

    output_dir = os.path.join(args.root_dir, 'bin')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    split = args.split if args.split != 'dev' else 'valid'
    vocab = Dictionary.load(os.path.join(args.roberta_dir, 'roberta.base', 'dict.txt'))
    dataset_builder = indexed_dataset.make_builder(
        os.path.join(output_dir, split+'.text.bin'),
        impl=args.dataset_impl,
        vocab_size=len(vocab),
    )
    relations_builder = indexed_dataset.make_builder(
        os.path.join(output_dir, split+'.relations.bin'),
        impl=args.dataset_impl,
        vocab_size=None,
    )
    processor.initializer()
    annotations_list = list()
    total_length, num_sentences = 0, 0
    for i in range(len(texts)):
        text, relation_type_id = [texts[i]], unique_relation_types.index(relation_types[i])
        for ids_tensor, _annotations_list in map(processor, text):
            dataset_builder.add_item(ids_tensor)
            relations_builder.add_item(torch.IntTensor([relation_type_id]))
            _annotations_list[:, 0] += total_length
            _annotations_list[:, 1] += total_length
            _annotations_list[:, 2] += num_sentences
            _annotations_list[:, 3] += num_sentences
            num_sentences += 1
            total_length += len(ids_tensor)
            annotations_list.append(_annotations_list)

            pbar.update()
    pbar.close()

    dataset_builder.finalize(os.path.join(output_dir, split+'.text.idx'))
    relations_builder.finalize(os.path.join(output_dir, split+'.relations.idx'))
    annotations_list = np.concatenate(annotations_list)
    np.save(os.path.join(output_dir, split+'.annotations'), annotations_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation for SemEval 2010 Task 8 dataset')
    parser.add_argument('--split', type=str, help='Dataset split', choices=['train', 'dev', 'test'])
    parser.add_argument('--root-dir', type=str, default='../data/SemEval2010_task8_all_data', help='SemEval 2010 Task 8 root directory')
    parser.add_argument('--roberta-dir', type=str, default='../data/roberta', help='RoBERTa directory with all dictionaries.')
    parser.add_argument('--append-eos', default=False, action='store_true')
    parser.add_argument(
        '--dataset-impl',
        metavar='FORMAT',
        default='mmap',
        choices=indexed_dataset.get_available_dataset_impl(),
        help='output dataset implementation')
    args = parser.parse_args()
    main(args)
