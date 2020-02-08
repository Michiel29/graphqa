import argparse
import json
import codecs
import copy
import numpy as np
import os
import tqdm

import torch

from fairseq.data import Dictionary
from fairseq.data import indexed_dataset
from fairseq.data.encoders.gpt2_bpe import get_encoder


TRAINING_TQDM_BAD_FORMAT = (
    '{l_bar}{bar}| '
    '{n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]'
)


class FewRelProcessor(object):
    def __init__(self, roberta_dir, dataset_impl, append_eos):
        self.dataset_impl = dataset_impl
        self.append_eos = append_eos
        self.roberta_dir = roberta_dir
        assert os.path.isdir(self.roberta_dir)

    def initializer(self):
        global bpe
        bpe = get_encoder(
            os.path.join(self.roberta_dir, 'encoder.json'),
            os.path.join(self.roberta_dir, 'vocab.bpe'),
        )
        global vocab
        vocab = Dictionary.load(os.path.join(self.roberta_dir, 'dict.txt'))

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
        text = (
            text.lower()
            .replace('( ', '(').replace(' )', ')')
            .replace(' :', ':').replace(': ', ':')
            .replace(' ,', ',').replace(' %', '%')
            .replace(' \'', '\'').replace('\' ', '\'')
            .replace(' ...', '...').replace(' / ', '/')
            .replace(' "', '"').replace('" ', '"')
            .replace(' !', '!').replace(' .', '.')
            .replace(' -', '-').replace('- ', '-')
            .replace('„ ', '„').replace(' “', '“')
            .replace(' \u2013 ', '\u2013').replace(' \u2014 ', '\u2014')
            .replace(' °', '°').replace(' ?', '?')
            .replace('# ', '#').replace('$ ', '$')
            .replace('¡ ', '¡').replace('. ', '.')
            .replace('[ ', '[').replace('¿ ', '¿')
            .replace('! ', '!').replace(' …', '…')
        )

        surface_form = (
            surface_form
            .replace(' "', '"').replace('" ', '"')
            .replace(' - ', '-').replace(' \u2013 ', '\u2013')
            .replace(' \'', '\'').replace('\' ', '\'')
            .replace('. ', '.').replace(' / ', '/')
            .replace(': ', ':').replace(' !', '!')
        )
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

    def build_annotations(self, d, entity_id):
        assert len(d) == 3
        return [
            {
                'surface_form': d[0],
                'start_word': min(interval),
                'end_word': max(interval) + 1,
                'uri': entity_id,
            }
            for interval in d[2]
        ]

    def __call__(self, sample):
        global vocab
        annotations = self.build_annotations(sample['h'], 0) + self.build_annotations(sample['t'], 1)
        annotations.sort(key=lambda x: x['start_word'])
        words, annotations = self.strip_empty_words(sample['tokens'], annotations)
        ids, annotations = self.apply_gt2_bpe(' '.join(words), annotations)
        ids_tensor = vocab.encode_line(line=' '.join(ids), append_eos=self.append_eos)

        assert len(ids_tensor) == len(ids) + int(self.append_eos)
        annotations_tensor = torch.IntTensor([
            [annotation['start_word'], annotation['end_word'], annotation['uri']]
            for annotation in annotations
        ])
        return ids_tensor, annotations_tensor


def main(args):
    with codecs.open(args.data, 'r', 'utf8') as f:
        data = json.load(f)
    print('-- Loaded data from %s' % args.data)

    processor = FewRelProcessor(
        args.roberta,
        args.dataset_impl,
        args.append_eos,
    )
    pbar = tqdm.tqdm(
        total=sum([len(v) for _, v in data.items()]),
        desc='Processing Wiki',
        bar_format=TRAINING_TQDM_BAD_FORMAT,
    )

    vocab = Dictionary.load('/data2/urikz/nki/roberta/dict.txt')
    dataset_builder = indexed_dataset.make_builder(
        args.output + '.text.bin',
        impl=args.dataset_impl,
        vocab_size=len(vocab),
    )
    annotations_builder = indexed_dataset.make_builder(
        args.output + '.annotations.bin',
        impl=args.dataset_impl,
        vocab_size=None,
    )
    relations_builder = indexed_dataset.make_builder(
        args.output + '.relations.bin',
        impl=args.dataset_impl,
        vocab_size=None,
    )
    processor.initializer()
    for relation_type_id, (_, samples) in enumerate(data.items()):
        for ids_tensor, annotations_tensor in map(processor, samples):
            dataset_builder.add_item(ids_tensor)
            annotations_builder.add_item(annotations_tensor)
            relations_builder.add_item(torch.IntTensor([relation_type_id]))
            pbar.update()
    pbar.close()

    dataset_builder.finalize(args.output + '.text.idx')
    annotations_builder.finalize(args.output + '.annotations.idx')
    relations_builder.finalize(args.output + '.relations.idx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation for FewRel dataset')
    parser.add_argument('--data', type=str, help='Input FewRel json file')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--roberta', type=str, help='RoBERTa directory with all dictionaries.')
    parser.add_argument('--append-eos', default=False, action='store_true')
    parser.add_argument(
        '--dataset-impl',
        metavar='FORMAT',
        default='mmap',
        choices=indexed_dataset.get_available_dataset_impl(),
        help='output dataset implementation')
    args = parser.parse_args()
    main(args)