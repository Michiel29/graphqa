import argparse
import spacy
from intervaltree import IntervalTree
import codecs
import multiprocessing
import glob
import numpy as np
import os
import tqdm

import torch

from fairseq.data import Dictionary
from fairseq.data import indexed_dataset
from fairseq.data.encoders.gpt2_bpe import get_encoder

from utils.dictionary import CustomDictionary


TRAINING_TQDM_BAD_FORMAT = (
    '{l_bar}{bar}| '
    '{n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]'
)


def get_start_end(mention):
    return mention['offset'], mention['offset'] + len(mention['surface_form'])


class FakeSpan(object):
    def __init__(self, text, start_char, end_char):
        self.text = str(text)
        self.start_char = start_char
        self.end_char = end_char

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


CUSTOM_RULES = {
    'Cruz Azul': 'Cruz Azul Fútbol Club'
}


CUSTOM_LENGTH_ANNOTATION = {
    'the He-Man/Masters': len('the He-Man'),
    'TACA Airlines - Monseñor Óscar': len('TACA Airlines'),
}


CUSTOM_FIX_ANSWER = {
    'Cao and Mao': 'Mao Zedong',
}


class WikiProcessor(object):
    def __init__(
        self,
        roberta_dir,
        dataset_impl,
        append_eos,
    ):
        self.roberta_dir = roberta_dir
        assert os.path.isdir(self.roberta_dir)
        self.dataset_impl = dataset_impl
        self.append_eos = append_eos
        self.nlp = spacy.load("en_core_web_lg")

    def initializer(self):
        global bpe
        bpe = get_encoder(
            os.path.join(self.roberta_dir, 'encoder.json'),
            os.path.join(self.roberta_dir, 'vocab.bpe'),
        )
        global vocab
        vocab = Dictionary.load(os.path.join(self.roberta_dir, 'dict.txt'))
        global custom_dictionary
        custom_dictionary = CustomDictionary.load(os.path.join(self.roberta_dir, 'dict.txt'))

    def find_all(self, a_str, sub):
        start = 0
        while True:
            start = a_str.find(sub, start)
            if start == -1: return
            yield start, start + len(sub)
            start += len(sub)

    def approx_find(self, text, entities, entity2id, xxx):
        matched_index = None
        for index, entity in enumerate(entities):
            if xxx in str(entity):
                if matched_index is None:
                    matched_index = index
                else:
                    if entity2id[str(entity)] != entity2id[str(entities[matched_index])] and len(str(entity)) < len(str(entities[matched_index])):
                        matched_index = index

        if matched_index is None:
            # print('WARNING: missing entity "%s" amongst entities %s' % (xxx, ','.join([str(e) for e in entities])))
            matched_index = []
            for start_pos, end_pos in self.find_all(text, xxx):
                matched_index.append(len(entities))
                entities.append(FakeSpan(xxx, start_pos, end_pos))
            assert len(matched_index) > 0
            return matched_index, entities

        assert matched_index is not None
        return [matched_index], entities

    def fix_entities(self, entities):
        for i in range(len(entities)):
            s_i = str(entities[i])
            if s_i in CUSTOM_LENGTH_ANNOTATION:
                new_len = CUSTOM_LENGTH_ANNOTATION[s_i]
                entities[i] = FakeSpan(
                    text=s_i[:new_len],
                    start_char=entities[i].start_char,
                    end_char=entities[i].start_char + new_len,
                )

        for i in range(len(entities)):
            s_i = str(entities[i])
            for j in range(len(entities)):
                s_j = str(entities[j])
                if (
                    i != j
                    and (s_i + '\'s' == s_j)
                ):
                    entities[j] = FakeSpan(
                        text=entities[i],
                        start_char=entities[j].start_char,
                        end_char=entities[j].end_char - 2,
                    )

    def normalize(self, s):
        return CUSTOM_RULES.get(s, s).lower()

    def build_index(self, entities):
        entitiy2id = {}
        index = 2
        for i in range(len(entities)):
            es = str(entities[i])
            found_similar_str = None
            for es2 in entitiy2id.keys():
                if self.normalize(es2) == self.normalize(es):
                    assert found_similar_str is None or entitiy2id[found_similar_str] == entitiy2id[es2]
                    found_similar_str = es2
            if found_similar_str is not None:
                entitiy2id[es] = entitiy2id[found_similar_str]
            else:
                entitiy2id[es] = index
                index += 1
        return entitiy2id

    def update_entity2id(self, entity2id, from_index, to_index):
        for k, v in entity2id.items():
            if v == from_index:
                entity2id[k] = to_index
        return entity2id

    def build_entities(self, text, xxx_entity, answer):
        entities = list(self.nlp(text).ents)
        self.fix_entities(entities)
        entity2id = self.build_index(entities)
        intervals = IntervalTree()
        xxx_index, entities = self.approx_find(text, entities, entity2id, xxx_entity)

        for idx in xxx_index:
            se = str(entities[idx])
            if se in entity2id and entity2id[se] != 0:
                entity2id = self.update_entity2id(entity2id, entity2id[se], 0)
            entity2id[se] = 0
            intervals.addi(entities[idx].start_char, entities[idx].end_char, 0)

        if answer is not None:
            answer_index, entities = self.approx_find(text, entities, entity2id, answer)
            for idx in answer_index:
                se = str(entities[idx])
                if se in entity2id and entity2id[se] != 1:
                    entity2id = self.update_entity2id(entity2id, entity2id[se], 1)
                entity2id[se] = 1
                intervals.addi(entities[idx].start_char, entities[idx].end_char, 1)

        entities = [
            entity
            for entity in entities
            if entity2id[str(entity)] in [0, 1] or not intervals.overlap(entity.start_char, entity.end_char)
        ]

        index = 2
        for entity in entities:
            if str(entity) not in entity2id:
                entity2id[str(entity)] = index
                index += 1
        return entities, entity2id

    def build_annotations(self, text, entities, entity2id):
        annotations = []
        for idx, entity in enumerate(entities):
            assert entity.start_char <= entity.end_char - 1
            annotations.append({
                'surface_form': str(entity),
                'start_char': entity.start_char,
                'end_char': entity.end_char,
                'uri': entity2id[str(entity)],
            })
        annotations.sort(key=lambda annotation: annotation['start_char'])
        for i in range(len(annotations)):
            start_pos = annotations[i]['start_char']
            if start_pos > 0 and text[start_pos - 1] != ' ':
                text = text[:start_pos] + ' ' + text[start_pos:]
                for j in range(i, len(annotations)):
                    annotations[j]['start_char'] += 1
                    annotations[j]['end_char'] += 1

            end_pos = annotations[i]['end_char']
            if end_pos < len(text) and text[end_pos] != ' ':
                text = text[:end_pos] + ' ' + text[end_pos:]
                for j in range(i + 1, len(annotations)):
                    annotations[j]['start_char'] += 1
                    annotations[j]['end_char'] += 1

        return text, self.get_word_based_offsets(text, annotations)

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
                        start = annotations[next_annotation_index]['start_char']
                        if i == start:
                            current_annotation_index = next_annotation_index
                            annotations[current_annotation_index]['start_word'] = word_index

                    if current_annotation_index is not None:
                        end = annotations[next_annotation_index]['end_char']
                        if i == end - 1:
                            annotations[current_annotation_index]['end_word'] = word_index + 1
                            next_annotation_index += 1
                            current_annotation_index = None

            words = sentence.split(' ')
            for annotation in annotations:
                if (
                    'start_word' not in annotation
                    or 'end_word' not in annotation
                    or annotation['surface_form'] != ' '.join(words[annotation['start_word']:annotation['end_word']])
                ):
                    return None
            return annotations

    def apply_gt2_bpe(self, sentence, annotations):
        global bpe
        ids = list(map(str, bpe.encode(sentence)))
        if len(annotations) == 0:
            return ids, annotations
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

    def process_question(self, question):
        global custom_dictionary
        if 'XXX' not in question.split(' '):
            start_pos = question.find('XXX')
            end_pos = start_pos + len('XXX')
            assert start_pos + len('XXX') < len(question)
            question = question[:end_pos] + ' ' + question[end_pos:]

        start_xxx = question.split(' ').index('XXX')
        question_annotations = [{
            'uri': 0,
            'start_word': start_xxx,
            'end_word': start_xxx + 1,
            'surface_form': 'XXX',
        }]
        question_ids, question_annotations = self.apply_gt2_bpe(question, question_annotations)
        ids_tensor = vocab.encode_line(line=' '.join(question_ids), append_eos=False).tolist()
        assert len(ids_tensor) == len(question_ids)
        assert len(question_annotations) == 1
        assert question_annotations[0]['start_word'] + 1 == question_annotations[0]['end_word']
        ids_tensor[question_annotations[0]['start_word']] = custom_dictionary.blank()

        question_annotations.append({
            'uri': 1,
            'start_word': len(ids_tensor),
            'end_word': len(ids_tensor) + 1,
        })
        ids_tensor.append(custom_dictionary.blank())

        if self.append_eos:
            ids_tensor.append(vocab.eos())
        ids_tensor = torch.tensor(ids_tensor, dtype=torch.int32)
        return ids_tensor, question_annotations

    def annotations_to_list(self, annotations):
        return np.array([
            [
                annotation['start_word'],
                annotation['end_word'],
                0,
                0,
                annotation['uri'],
            ]
            for annotation in annotations
        ], dtype=np.int64)


    def __call__(self, path):
        global vocab

        with codecs.open(path, 'r', 'utf8') as f:
            for line in f:
                columns = line[:-1].split('\t')
                if len(columns) < 5:
                    _, question, xxx_entity, evidence = columns
                    answer = None
                else:
                    # TODO: Handle more than one answer
                    _, question, xxx_entity, evidence, answer = columns[:5]
                    answer = CUSTOM_FIX_ANSWER.get(answer, answer)

                entities, entity2id = self.build_entities(evidence, xxx_entity, answer)
                text, annotations = self.build_annotations(evidence, entities, entity2id)
                if annotations is None:
                    continue

                if len(annotations) > 40:
                    blabla = 1
                    pass

                ids, annotations = self.apply_gt2_bpe(text, annotations)
                ids_tensor = vocab.encode_line(line=' '.join(ids), append_eos=self.append_eos)
                assert len(ids_tensor) == len(ids) + int(self.append_eos)

                quesiton_ids_tensor, question_annotations = self.process_question(question)

                yield (
                    quesiton_ids_tensor,
                    self.annotations_to_list(question_annotations),
                    ids_tensor,
                    self.annotations_to_list(annotations),
                )

def main(args):
    input_files = sorted([
        path
        for data in args.data.split(',')
        for path in glob.glob(os.path.expanduser(data))
    ])
    print('-- Found %d files' % len(input_files))

    processor = WikiProcessor(
        args.roberta,
        args.dataset_impl,
        args.append_eos,
    )
    processor.initializer()

    pbar = tqdm.tqdm(
        total=len(input_files),
        desc='Processing QA-ZRE',
        bar_format=TRAINING_TQDM_BAD_FORMAT,
    )

    custom_dictionary = CustomDictionary.load(os.path.join(args.roberta, 'dict.txt'))
    dataset_builder = indexed_dataset.make_builder(
        args.output + '.text.bin',
        impl=args.dataset_impl,
        vocab_size=len(custom_dictionary),
    )

    num_samples, num_sentences, total_length = 0, 0, 0
    num_unique_entities_sum, num_unique_entities_min, num_unique_entities_max = 0, None, None

    annotations_list = []

    for annotated_tensors in map(processor, input_files):
        for question_tensor, question_annotations, text_tensor, text_annotations in annotated_tensors:
            # extra checks
            assert question_annotations.shape == (2, 5)
            assert question_tensor[question_annotations[0, 0]:question_annotations[0, 1]].item() == custom_dictionary.blank()
            assert question_tensor[question_annotations[1, 0]:question_annotations[1, 1]].item() == custom_dictionary.blank()

            question_annotations[:, 0] += total_length
            question_annotations[:, 1] += total_length
            question_annotations[:, 2] += num_sentences
            question_annotations[:, 3] += num_samples

            dataset_builder.add_item(question_tensor)
            annotations_list.append(question_annotations)
            total_length += len(question_tensor)
            num_sentences += 1

            text_annotations[:, 0] += total_length
            text_annotations[:, 1] += total_length
            text_annotations[:, 2] += num_sentences
            text_annotations[:, 3] += num_samples

            dataset_builder.add_item(text_tensor)
            annotations_list.append(text_annotations)
            total_length += len(text_tensor)
            num_sentences += 1
            num_samples += 1

            num_unique_entities_current = len(np.unique(np.concatenate([text_annotations[:, 4], question_annotations[:, 4]])))
            num_unique_entities_sum += num_unique_entities_current
            num_unique_entities_min = min(num_unique_entities_min or num_unique_entities_current, num_unique_entities_current)
            num_unique_entities_max = max(num_unique_entities_max or num_unique_entities_current, num_unique_entities_current)

            pbar.set_postfix({
                's': num_samples,
                'ue_avg': num_unique_entities_sum / num_samples,
                'ue_min': num_unique_entities_min,
                'ue_max': num_unique_entities_max,
            })
        pbar.update()

    dataset_builder.finalize(args.output + '.text.idx')
    np.save(args.output + '.annotations', np.concatenate(annotations_list))
    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation for Wiki Slot-Filling zeroshot dataset')
    parser.add_argument('--data', type=str, help='Input files pattern')
    parser.add_argument('--output', type=str, help='Output files prefix')
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
