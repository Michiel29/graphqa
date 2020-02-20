import argparse
import json
import codecs
import copy
from collections import Counter
import glob
import multiprocessing as mp
import numpy as np
import os
import sys
import tempfile
import tqdm
from ncls import NCLS
from nltk.tokenize.punkt import PunktSentenceTokenizer

import torch

from fairseq.data import Dictionary
from fairseq.data import indexed_dataset
from fairseq.data.encoders.gpt2_bpe import get_encoder

TRAINING_TQDM_BAD_FORMAT = (
    '{l_bar}{bar}| '
    '{n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]'
)


def get_start_end(mention):
    return mention['offset'], mention['offset'] + len(mention['surface_form'])


def intersect(s1, e1, s2, e2):
    return (s1 <= s2 and s2 < e1) or (s2 <= s1 and s1 < e2)


def get_intervals(annotations):
    start, end = [], []
    for annotation in annotations:
        s, e = get_start_end(annotation)
        start.append(s)
        end.append(e)
    return np.array(start, dtype=np.int64), np.array(end, dtype=np.int64), np.arange(len(annotations))


def load_entities(path):
    assert os.path.exists(path)
    entities = {}
    with codecs.open(path, 'r', 'utf-8') as f:
        for index, line in enumerate(f):
            entity, counts = line[:-1].split(' ')
            entities[entity] = int(index)
    return entities


class WikiProcessor(object):
    def __init__(
        self,
        roberta_dir,
        limit_set_of_entities,
        dataset_impl,
        tmp,
        append_eos,
        entity_vocab,
    ):
        self.roberta_dir = roberta_dir
        assert os.path.isdir(self.roberta_dir)
        self.limit_set_of_entities = limit_set_of_entities
        self.dataset_impl = dataset_impl
        self.tmp = tmp
        self.append_eos = append_eos
        self.entity_vocab = entity_vocab

    def initializer(self):
        global bpe
        bpe = get_encoder(
            os.path.join(self.roberta_dir, 'encoder.json'),
            os.path.join(self.roberta_dir, 'vocab.bpe'),
        )
        global vocab
        vocab = Dictionary.load(os.path.join(self.roberta_dir, 'dict.txt'))
        global entities
        if self.entity_vocab is not None:
            entities = load_entities(self.entity_vocab)

    def generate_tmp_filename(self):
        tf = tempfile.NamedTemporaryFile(
            dir=self.tmp,
            delete=True,
        )
        tf.close()
        return tf.name

    def fix_annotations(self, annotations):
        new_annotations = []
        num_filtered = 0
        for annotation in annotations:
            if u'\xa0' in annotation['surface_form'] or '\n' in annotation['uri']:
                num_filtered += 1
                continue
            while annotation['surface_form'].startswith(' '):
                annotation['surface_form'] = annotation['surface_form'][1:]
                annotation['offset'] += 1
            while annotation['surface_form'].endswith(' '):
                annotation['surface_form'] = annotation['surface_form'][:-1]
            assert len(annotation['surface_form']) > 0
            new_annotations.append(annotation)
        return new_annotations, num_filtered

    def filter_by_candidate_set(self, article, annotations):
        if not self.limit_set_of_entities:
            return annotations, 0

        main_entity = article['url'][len('http://en.wikipedia.org/wiki/'):]
        original_entities_set = set([original_entity['uri'] for original_entity in article['annotations']])
        original_entities_set.add(main_entity)

        new_annotations = []
        num_filtered = 0
        for annotation in annotations:
            if annotation['uri'] in original_entities_set:
                new_annotations.append(annotation)
            else:
                num_filtered += 1
        assert len(new_annotations) + num_filtered == len(annotations)
        return new_annotations, num_filtered

    def filter_by_human_annotations(self, article, annotations):
        ncls = NCLS(*get_intervals(article['annotations']))
        new_annotations = []
        num_filtered = 0
        for annotation in annotations:
            entity_start, entity_end = get_start_end(annotation)
            matched_human_annotation = list(ncls.find_overlap(entity_start, entity_end))
            if len(matched_human_annotation) == 0:
                new_annotations.append(annotation)
            else:
                human_annotation = article['annotations'][matched_human_annotation[0][2]]
                human_annotation_start, human_annotation_end = get_start_end(human_annotation)
                assert intersect(human_annotation_start, human_annotation_end, entity_start, entity_end)
                num_filtered += 1
        assert len(new_annotations) + num_filtered == len(annotations)
        return new_annotations, num_filtered

    def filter_by_self_overlaps(self, annotations):
        new_annotations = []
        num_filtered = 0
        start, end, index = get_intervals(annotations)
        events = sorted([(x[0], 1, x[1]) for x in zip(start, index)] + [(x[0], 0, x[1]) for x in zip(end, index)])
        current_open_interval = None
        for event in events:
            if event[1] == 0: # close interval
                if event[2] == current_open_interval:
                    current_open_interval = None
            else:
                assert event[1] == 1 # open interval
                if current_open_interval is None:
                    current_open_interval = event[2]
                    new_annotations.append(annotations[current_open_interval])
                else:
                    num_filtered += 1
        assert len(new_annotations) + num_filtered == len(annotations)
        # TODO: Only for debugging
        # for i in range(len(new_annotations)):
        #     start_i, end_i = get_start_end(new_annotations[i])
        #     for j in range(i + 1, len(new_annotations)):
        #         start_j, end_j = get_start_end(new_annotations[j])
        #         assert not intersect(start_i, end_i, start_j, end_j)
        return new_annotations, num_filtered

    def filter_by_entity_vocab(self, annotations):
        global entities
        assert entities is not None
        new_annotations = []
        num_filtered = 0
        for annotation in annotations:
            if annotation['uri'] in entities:
                new_annotations.append(annotation)
            else:
                num_filtered += 1
        return new_annotations, num_filtered

    def split_into_sentences(self, text):
        offset = 0
        for paragraph in text.split('\n'):
            spans = list(PunktSentenceTokenizer().span_tokenize(paragraph))
            for i in range(len(spans)):
                start, end = spans[i]
                yield paragraph[start:end], offset + start
            offset += len(paragraph) + 1
        assert offset == len(text) + 1

    def set_local_offsets(self, offset, annotations):
        for annotation in annotations:
            annotation['offset'] -= offset
        return annotations

    def add_margin_to_annotations(self, sentence, annotations):
        sorted_annotations = sorted(annotations, key=lambda x: x['offset'])
        for i in range(len(sorted_annotations)):
            start, end = get_start_end(sorted_annotations[i])
            assert sentence[start:end] == sorted_annotations[i]['surface_form']
            if start > 0 and sentence[start - 1] != ' ':
                sentence = sentence[:start] + ' ' + sentence[start:]
                for j in range(i, len(sorted_annotations)):
                    sorted_annotations[j]['offset'] += 1
                start, end = get_start_end(sorted_annotations[i])

            if end < len(sentence) and sentence[end] != ' ':
                sentence = sentence[:end] + ' ' + sentence[end:]
                for j in range(i + 1, len(sorted_annotations)):
                    sorted_annotations[j]['offset'] += 1
                    start, end = get_start_end(sorted_annotations[j])
                    assert sentence[start:end] == sorted_annotations[j]['surface_form']

        assert sentence.find('  ') == -1
        return sentence, sorted_annotations

    def strip_whitespaces(self, sentence, annotations):
        sentence = sentence.rstrip()
        num_whitespaces = 0
        while num_whitespaces < len(sentence) and sentence[num_whitespaces] == ' ':
            num_whitespaces += 1
        assert num_whitespaces < len(sentence)
        if num_whitespaces > 0:
            sentence = sentence[num_whitespaces:]
            for i in range(len(annotations)):
                annotations[i]['offset'] -= num_whitespaces
        return sentence, annotations

    def strip_double_whitespaces(self, sentence, annotations):
        idx = sentence.find('  ')
        while idx >= 0:
            sentence = sentence[:idx] + sentence[idx + 1:]
            for i in range(len(annotations)):
                assert annotations[i]['offset'] != idx
                if annotations[i]['offset'] > idx:
                    annotations[i]['offset'] -= 1
                start, end = get_start_end(annotations[i])
                assert sentence[start:end] == annotations[i]['surface_form']
            idx = sentence.find('  ')
        return sentence, annotations

    def get_word_based_offsets(self, sentence, annotations):
        word_index = 0
        current_annotation_index, next_annotation_index = None, 0
        assert sentence[0] != ' ', sentence
        for i in range(len(sentence)):
            if next_annotation_index >= len(annotations):
                break
            if sentence[i] == ' ':
                word_index += 1
            else:
                if current_annotation_index is None:
                    start, _ = get_start_end(annotations[next_annotation_index])
                    if i == start:
                        current_annotation_index = next_annotation_index
                        annotations[current_annotation_index]['start_word'] = word_index

                if current_annotation_index is not None:
                    _, end = get_start_end(annotations[current_annotation_index])
                    if i == end - 1:
                        annotations[current_annotation_index]['end_word'] = word_index + 1
                        next_annotation_index += 1
                        current_annotation_index = None

        words = sentence.split(' ')
        for annotation in annotations:
            assert annotation['surface_form'] == ' '.join(words[annotation['start_word']:annotation['end_word']])
        return annotations

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
            if decoded.startswith(' '):
                decoded = decoded[1:]
            assert annotation['surface_form'] == decoded
        return ids, annotations


    def __call__(self, path):
        global vocab
        global entities
        num_annotations, num_sentences = 0, 0
        num_filtered_xao = 0
        num_filtered_by_candidate_set, num_filtered_by_human_annotations, num_filtered_by_self_overlaps = 0, 0, 0
        num_filtered_by_entity_vocab = 0

        if self.entity_vocab is None:
            annotation_entities = Counter()
        else:
            output_prefix = self.generate_tmp_filename()
            dataset_builder = indexed_dataset.make_builder(
                output_prefix + '.text.bin',
                impl=self.dataset_impl,
                vocab_size=len(vocab),
            )
            annotations_builder = indexed_dataset.make_builder(
                output_prefix + '.annotations.bin',
                impl=self.dataset_impl,
                vocab_size=len(entities),
            )

        with codecs.open(path, 'r', 'utf8') as f:
            for line in f:
                article = json.loads(line[:-1])
                annotations = article['el']
                article['annotations'], _num_filtered_xao = self.fix_annotations(article['annotations'])
                num_filtered_xao += _num_filtered_xao
                annotations, _num_filtered_xao = self.fix_annotations(annotations)
                num_filtered_xao += _num_filtered_xao
                annotations, _num_filtered_by_candidate_set = self.filter_by_candidate_set(article, annotations)
                annotations, _num_filtered_by_human_annotations = self.filter_by_human_annotations(article, annotations)
                annotations, _num_filtered_by_self_overlaps = self.filter_by_self_overlaps(annotations)
                annotations = article['annotations'] + annotations
                if self.entity_vocab is not None:
                    annotations, _num_filtered_by_entity_vocab = self.filter_by_entity_vocab(annotations)
                    num_filtered_by_entity_vocab += _num_filtered_by_entity_vocab
                num_filtered_by_candidate_set += _num_filtered_by_candidate_set
                num_filtered_by_human_annotations += _num_filtered_by_human_annotations
                num_filtered_by_self_overlaps += _num_filtered_by_self_overlaps

                nlcs = NCLS(*get_intervals(annotations))
                text = article['text'].replace(u'\xa0', u' ')
                offset = 0
                num_filtered_by_crossing_sentence_boundaries, num_filtered_solo_annotion_in_sentence = 0, 0

                for sentence, offset in self.split_into_sentences(text):
                    sentence_begin = offset
                    sentence_end = offset + len(sentence)
                    assert sentence == text[sentence_begin:sentence_end]

                    annotations_per_sentence = []
                    for annotation_id in nlcs.find_overlap(sentence_begin, sentence_end):
                        annotation = annotations[annotation_id[2]]
                        start, end = get_start_end(annotation)
                        if sentence_begin <= start and end <= sentence_end:
                            annotations_per_sentence.append(annotation)
                        else:
                            num_filtered_by_crossing_sentence_boundaries += 1
                    if len(annotations_per_sentence) == 0:
                        continue
                    if len(set([annotation['uri'] for annotation in annotations_per_sentence])) < 2:
                        num_filtered_solo_annotion_in_sentence += 1
                        continue
                    num_annotations += len(annotations_per_sentence)
                    num_sentences += 1

                    if self.entity_vocab is None:
                        annotation_entities.update([annotation['uri'] for annotation in annotations_per_sentence])
                    else:
                        annotations_per_sentence = self.set_local_offsets(offset, annotations_per_sentence)
                        fixed_sentence, annotations_per_sentence = self.strip_whitespaces(sentence, annotations_per_sentence)
                        fixed_sentence, annotations_per_sentence = self.strip_double_whitespaces(fixed_sentence, annotations_per_sentence)
                        fixed_sentence, annotations_per_sentence = self.add_margin_to_annotations(fixed_sentence, annotations_per_sentence)
                        annotations_per_sentence = self.get_word_based_offsets(fixed_sentence, annotations_per_sentence)
                        ids, annotations_per_sentence = self.apply_gt2_bpe(fixed_sentence, annotations_per_sentence)

                        ids_tensor = vocab.encode_line(line=' '.join(ids), append_eos=self.append_eos)
                        assert len(ids_tensor) == len(ids) + 1
                        dataset_builder.add_item(ids_tensor)
                        annotations_builder.add_item(torch.IntTensor(
                            [[x['start_word'], x['end_word'], int(entities[x['uri']])] for x in annotations_per_sentence]
                        ))

        if self.entity_vocab is not None:
            dataset_builder.finalize(output_prefix + '.text.idx')
            annotations_builder.finalize(output_prefix + '.annotations.idx')
        return (
            annotation_entities if self.entity_vocab is None else output_prefix,
            num_sentences,
            num_annotations,
            num_filtered_by_candidate_set,
            num_filtered_by_human_annotations,
            num_filtered_by_self_overlaps,
            num_filtered_by_crossing_sentence_boundaries,
            num_filtered_solo_annotion_in_sentence,
            num_filtered_xao,
            num_filtered_by_entity_vocab,
        )


def main(args):
    assert os.path.isdir(args.tmp)
    input_files = sorted([
        path
        for data in args.data.split(',')
        for path in glob.glob(os.path.expanduser(data))
    ])
    print('-- Found %d files' % len(input_files))
    build_entity_vocab_mode = not os.path.exists(args.entity_vocab)
    print('-- Build entity vocab mode: %s' % ('ON' if build_entity_vocab_mode else 'OFF'))

    processor = WikiProcessor(
        args.roberta,
        args.limit_set_of_entities,
        args.dataset_impl,
        args.tmp,
        args.append_eos,
        args.entity_vocab if not build_entity_vocab_mode else None,
    )
    num_sentences = 0
    num_annotations, num_filtered_by_candidate_set, num_filtered_by_human_annotations, num_filtered_by_self_overlaps = 0, 0, 0, 0
    num_filtered_xao, num_filtered_by_crossing_sentence_boundaries, num_filtered_solo_annotion_in_sentence = 0, 0, 0
    num_filtered_by_entity_vocab = 0

    pbar = tqdm.tqdm(
        total=len(input_files),
        desc='Processing Wiki',
        bar_format=TRAINING_TQDM_BAD_FORMAT,
    )
    pbar.set_postfix({
        's': num_sentences,
        'ann': num_annotations,
        'f_ed': num_filtered_by_candidate_set,
        'f_h_overlap': num_filtered_by_human_annotations,
        'f_self_overlap': num_filtered_by_self_overlaps,
        'f_cross_s_bd': num_filtered_by_crossing_sentence_boundaries,
        'f_solo_s': num_filtered_solo_annotion_in_sentence,
        'f_xao': num_filtered_xao,
        'f_vocab': num_filtered_by_entity_vocab,
    })

    if build_entity_vocab_mode:
        entities = Counter()
    else:
        vocab = Dictionary.load('/data/urikz/nki/roberta/dict.txt')
        dataset_builder = indexed_dataset.make_builder(
            args.output + '.text.bin',
            impl=args.dataset_impl,
            vocab_size=len(vocab),
        )
        entities = load_entities(args.entity_vocab)
        annotations_builder = indexed_dataset.make_builder(
            args.output + '.annotations.bin',
            impl=args.dataset_impl,
            vocab_size=len(entities),
        )
    if args.nworkers == 1:
        processor.initializer()
        for output, s, x, y, z, w, v, u, t, q in map(processor, input_files):
            if build_entity_vocab_mode:
                entities.update(output)
            else:
                dataset_builder.merge_file_(output + '.text')
                annotations_builder.merge_file_(output + '.annotations')
            num_sentences += s
            num_annotations += x
            num_filtered_by_candidate_set += y
            num_filtered_by_human_annotations += z
            num_filtered_by_self_overlaps += w
            num_filtered_by_crossing_sentence_boundaries += v
            num_filtered_solo_annotion_in_sentence += u
            num_filtered_xao += t
            num_filtered_by_entity_vocab += q
            pbar.set_postfix({
                's': num_sentences,
                'ann': num_annotations,
                'f_ed': num_filtered_by_candidate_set,
                'f_h_overlap': num_filtered_by_human_annotations,
                'f_self_overlap': num_filtered_by_self_overlaps,
                'f_cross_s_bd': num_filtered_by_crossing_sentence_boundaries,
                'f_solo_s': num_filtered_solo_annotion_in_sentence,
                'f_xao': num_filtered_xao,
                'f_vocab': num_filtered_by_entity_vocab,
            })
            pbar.update()
    else:
        with mp.Pool(processes=args.nworkers, initializer=processor.initializer) as pool:
            for output, s, x, y, z, w, v, u, t, q in pool.imap_unordered(processor, input_files):
                if build_entity_vocab_mode:
                    entities.update(output)
                else:
                    dataset_builder.merge_file_(output + '.text')
                    annotations_builder.merge_file_(output + '.annotations')
                num_sentences += s
                num_annotations += x
                num_filtered_by_candidate_set += y
                num_filtered_by_human_annotations += z
                num_filtered_by_self_overlaps += w
                num_filtered_by_crossing_sentence_boundaries += v
                num_filtered_solo_annotion_in_sentence += u
                num_filtered_xao += t
                num_filtered_by_entity_vocab += q
                pbar.set_postfix({
                    's': num_sentences,
                    'ann': num_annotations,
                    'f_ed': num_filtered_by_candidate_set,
                    'f_h_overlap': num_filtered_by_human_annotations,
                    'f_self_overlap': num_filtered_by_self_overlaps,
                    'f_cross_s_bd': num_filtered_by_crossing_sentence_boundaries,
                    'f_solo_s': num_filtered_solo_annotion_in_sentence,
                    'f_xao': num_filtered_xao,
                    'f_vocab': num_filtered_by_entity_vocab,
                })
                pbar.update()
    pbar.close()

    if build_entity_vocab_mode:
        counter = 0
        with codecs.open(args.entity_vocab, 'w', 'utf8') as f:
            for entity_and_count in entities.most_common():
                if (
                    args.entity_count_threshold is None
                    or entity_and_count[1] >= args.entity_count_threshold
                ):
                    counter += 1
                    f.write('%s %d\n' % (entity_and_count[0], entity_and_count[1]))
        print('-- Successfully saved %d entities (out of %d) to %s' % (
            counter,
            len(entities),
            args.entity_vocab,
        ))
    else:
        dataset_builder.finalize(args.output + '.text.idx')
        annotations_builder.finalize(args.output + '.annotations.idx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation for Wiki')
    parser.add_argument('--data', type=str, help='Input files pattern')
    parser.add_argument(
        '--tmp',
        type=str,
        default='/tmp',
        help='Directory for temporary files',
    )
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--entity-vocab', type=str, default=None, help='Output filename')
    parser.add_argument('--limit-set-of-entities', default=False, action='store_true')
    parser.add_argument('--entity-count-threshold', default=None, type=int)
    parser.add_argument('--roberta', type=str, help='RoBERTa directory with all dictionaries.')
    parser.add_argument('--append-eos', default=False, action='store_true')
    parser.add_argument(
        '--dataset-impl',
        metavar='FORMAT',
        default='mmap',
        choices=indexed_dataset.get_available_dataset_impl(),
        help='output dataset implementation')
    parser.add_argument(
        '--nworkers',
        type=int,
        default=1,
        help='Number of workers',
    )
    args = parser.parse_args()
    main(args)
