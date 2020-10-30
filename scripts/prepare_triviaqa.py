import argparse
import json
import codecs
import copy
import numpy as np
import os
import tqdm
import re

import torch

from fairseq.data import Dictionary
from fairseq.data import indexed_dataset
from fairseq.data.encoders.gpt2_bpe import get_encoder

import sys; sys.path.append(os.path.join(sys.path[0], '..'))
from utils.dictionary import EntityDictionary

TRAINING_TQDM_BAD_FORMAT = (
    '{l_bar}{bar}| '
    '{n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]'
)


class TriviaQAProcessor(object):
    def __init__(self, roberta_dir, entity_path, dataset_impl, append_eos, max_positions):
        self.dataset_impl = dataset_impl
        self.append_eos = append_eos
        self.roberta_dir = roberta_dir
        assert os.path.isdir(self.roberta_dir)
        self.entity_path = entity_path

        self.max_positions = max_positions

    def initializer(self):
        global bpe
        bpe = get_encoder(
            os.path.join(self.roberta_dir, 'gpt2_bpe', 'encoder.json'),
            os.path.join(self.roberta_dir, 'gpt2_bpe', 'vocab.bpe'),
        )
        global vocab
        vocab = Dictionary.load(os.path.join(self.roberta_dir, 'roberta.base', 'dict.txt'))

        a = 3


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

        try:
            assert text == surface_form
        except:
            print('stop')

    def strip_empty_words(self, string):
        words = string.split(' ')
        new_words = []

        for word_id, word in enumerate(words):
            if len(word.strip()) != 0:
                new_words.append(word)
        new_string = ' '.join(new_words)
        return new_words, new_string

    def apply_gt2_bpe(self, string, annotation):
        global bpe
        ids = list(map(str, bpe.encode(string)))
        word_to_token = list()
        word_index = 0
        word_start = 0
        for token_id, token in enumerate(ids):
            if bpe.decode([int(token)]).startswith(' '):
                word_end = max(token_id, 0)
                word_to_token.append((word_start, word_end))
                word_start = token_id
                word_index += 1
        word_to_token.append((word_start, len(ids)-1))

        return ids, word_to_token

    def add_spaces_around_annotations(self, string, annotation):
        new_annotation = []

        length_sort_indices = np.argsort([len(entity['name']) for entity in annotation])[::-1]
        sorted_annotation = [annotation[i] for i in length_sort_indices]
        for entity in sorted_annotation:
            new_entity = copy.deepcopy(entity)
            if entity['metadata'] is not None and 'year' in entity['metadata']:
                continue

            surface_form = entity['mentions'][0]['content']
            if not surface_form in string:
                continue
            entity_start = string.index(surface_form)
            entity_end = entity_start + len(surface_form) - 1
            if entity_start > 0:
                if string[entity_start - 1] != ' ':
                    # Filter out cases where google erroniously recongizes entity in the middle of the word Eg: one in anyone
                    if entity_start - 2 > 0 and string[entity_start - 2] != ' ' and string[entity_start - 3] != ' ':
                        continue
                    string = string[:entity_start] + ' ' + string[entity_start:]
                    entity_end += 1

            if entity['type'] == 'DATE' and entity_end + 1 < len(string):
                if string[entity_end + 1] == 's':
                    new_entity['mentions'][0]['content'] = surface_form + 's'
                    entity_end += 1

            if entity_end + 1 < len(string):
                if string[entity_end + 1] != ' ':
                    if entity_end + 3 < len(string) and string[entity_end + 2] != ' ' and string[entity_end + 3] != ' ':
                        continue
                    string = string[:entity_end + 1] + ' ' + string[entity_end + 1:]


            new_annotation.append(new_entity)
        return string, new_annotation

    def find_all(self, substring, string):
        starts = []
        current_position = 0
        while True:
            position = string[current_position:].find(substring)
            if position >= 0:
                position = position + current_position
                starts.append(position)
                current_position = position + 1
            else:
                break
        return starts

    def process_annotation(self, string, words, annotation, word_to_token):
        strip_string = string.replace(' ', '')
        word_starts = []
        word_ends = []
        cumulative_characters = 0
        for word in words:
            word_starts.append(cumulative_characters)
            word_ends.append(cumulative_characters + len(word))
            cumulative_characters += len(word)
        already_done = []
        for entity in annotation:
            surface_form = entity['mentions'][0]['content']
            surface_strip = surface_form.replace(' ', '')
            candidate_character_positions = self.find_all(surface_strip, strip_string)
            candidate_word_positions = []
            for character_position in candidate_character_positions:
                if character_position in word_starts and character_position + len(surface_strip) in word_ends:
                    word_position = (word_starts.index(character_position), word_ends.index(character_position + len(surface_strip)))
                    candidate_word_positions.append(word_position)

            assert len(candidate_word_positions) > 0

            word_position = None
            for candidate_position in candidate_word_positions:
                if candidate_position not in already_done:
                    word_position = candidate_position
                    already_done.append(word_position)
                    break

            if word_position is None:
                continue

            start_token = word_to_token[word_position[0]]
            end_token = word_to_token[word_position[1]]
            entity['position'] = (start_token, end_token)
        return annotation

    def process(self, string, annotation):
        global vocab
        string, annotation = self.add_spaces_around_annotations(string, annotation)
        words, string = self.strip_empty_words(string)
        ids, word_to_token = self.apply_gt2_bpe(' '.join(words), annotation)
        annotation = self.process_annotation(string, words, annotation, word_to_token)

        if len(ids) >= self.max_positions:
            return None, None
        ids_tensor = vocab.encode_line(line=' '.join(ids), append_eos=self.append_eos)

        assert len(ids_tensor) == len(ids) + int(self.append_eos)
        return ids_tensor, annotation


def main(args):
    data_path = os.path.join(args.root_dir, args.split+'.json')
    annotation_path = os.path.join(args.root_dir, args.split+'_annotations.json')
    with codecs.open(data_path, 'r', 'utf8') as f:
        data = json.load(f)["Data"]

    with codecs.open(annotation_path, 'r', 'utf8') as f:
        annotations = json.load(f)
    print('-- Loaded data from %s' % data_path)

    processor = TriviaQAProcessor(
        args.roberta_dir,
        args.entity_path,
        args.dataset_impl,
        args.append_eos,
        args.max_positions
    )
    pbar = tqdm.tqdm(
        total=len(data),
        desc='Processing Wiki',
        bar_format=TRAINING_TQDM_BAD_FORMAT,
    )

    output_dir = os.path.join(args.root_dir, 'bin')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    split = args.split if args.split != 'dev' else 'valid'
    vocab = Dictionary.load(os.path.join(args.roberta_dir, 'roberta.base', 'dict.txt'))
    entity_dict = EntityDictionary.load(args.entity_path)
    question_builder = indexed_dataset.make_builder(
        os.path.join(output_dir, split + '.questions_entities' + '.text.bin'),
        impl=args.dataset_impl,
        vocab_size=len(vocab),
    )
    answer_entities = []
    processed_annotations = []

    processor.initializer()
    num_questions, valid_questions, no_annotation, not_in_dict, processing_problem = 0, 0, 0, 0, 0
    for i,sample in enumerate(data):
        num_questions += 1
        pbar.update()

        question = sample["Question"]
        answer = sample["Answer"]
        annotation = copy.deepcopy(annotations[i])

        entity_name = None

        # Only use questions with an entity for the answer
        if "MatchedWikiEntityName" in answer:
            entity_name = answer["MatchedWikiEntityName"]
        else:
            continue

        if annotation is None:
            no_annotation += 1
            continue

        entity_name = entity_name.replace(' ', '_')

        # assert entity_name in entity_dict
        if not entity_name in entity_dict:
            not_in_dict += 1
            continue
        entity_id = entity_dict.index(entity_name)


        ids_tensor, processed_annotation = processor.process(question, annotation)
        if ids_tensor is None:
            processing_problem += 1
            continue
        processed_annotations.append(processed_annotation)
        answer_entities.append(entity_id)


        question_builder.add_item(ids_tensor)
        valid_questions += 1

    pbar.close()

    question_builder.finalize(os.path.join(output_dir, split+ '.questions_entities' + '.idx'))
    np.save(os.path.join(output_dir, split+'.answer_entities'), answer_entities)

    processed_annotation_path = os.path.join(args.root_dir, args.split+'_processed_annotations.json')
    with codecs.open(processed_annotation_path, 'w', 'utf8') as f:
        json.dump(processed_annotations, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation for TriviaQA dataset')
    parser.add_argument('--split', type=str, help='Dataset split')
    parser.add_argument('--root-dir', type=str, default='../data/triviaqa/triviaqa/', help='TriviaQA root directory')
    parser.add_argument('--roberta-dir', type=str, default='../data/roberta', help='RoBERTa directory with all dictionaries.')
    parser.add_argument('--entity-path', type=str, default='../data/graphqa/entity.dict.txt', help='Txt file with dictionary of Wikipedia entities')
    parser.add_argument('--append-eos', default=False, action='store_true')
    parser.add_argument(
        '--dataset-impl',
        metavar='FORMAT',
        default='mmap',
        choices=indexed_dataset.get_available_dataset_impl(),
        help='output dataset implementation')
    parser.add_argument('--max-positions', type=int, default=123)
    args = parser.parse_args()
    main(args)
