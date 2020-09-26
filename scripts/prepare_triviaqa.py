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
        return new_words

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

        for entity in annotation:
            new_entity = copy.deepcopy(entity)
            if entity['metadata'] is not None and 'year' in entity['metadata']:
                continue

            surface_form = entity['mentions'][0]['content']

            entity_start = string.index(surface_form)
            entity_end = entity_start + len(surface_form) - 1
            if entity_start > 0:
                if string[entity_start - 1] != ' ':
                    string = string[:entity_start] + ' ' + string[entity_start:]
                    entity_end += 1

            if entity['type'] == 'DATE' and entity_end + 1 < len(string):
                if string[entity_end + 1] == 's':
                    new_entity['mentions'][0]['content'] = surface_form + 's'
                    entity_end += 1

            if entity_end + 1 < len(string):
                if string[entity_end + 1] != ' ':
                    string = string[:entity_end + 1] + ' ' + string[entity_end + 1:]

            new_annotation.append(new_entity)
        return string, new_annotation

    def process_annotation(self, words, annotation, word_to_token):
        already_done = []
        for entity in annotation:
            surface_form = entity['mentions'][0]['content']
            entity_words = surface_form.split(' ')
            candidate_starts = list()
            for i in range(len(words) - len(entity_words)):
                if entity_words == words[i:i+len(entity_words)]:
                    candidate_starts.append(i)

            assert len(candidate_starts) > 0

            position = None
            for candidate_start in candidate_starts:
                interval = (candidate_start, candidate_start + len(entity_words))
                if interval not in already_done:
                    position = interval
                    already_done.append(position)
                    break

            if position is None:
                continue

            start_token = word_to_token[position[0]]
            end_token = word_to_token[position[1]]
            entity['position'] = (start_token, end_token)
        return annotation

    def process(self, string, annotation):
        global vocab
        string, annotation = self.add_spaces_around_annotations(string, annotation)
        words = self.strip_empty_words(string)
        ids, word_to_token = self.apply_gt2_bpe(' '.join(words), annotation)
        annotation = self.process_annotation(words, annotation, word_to_token)

        if len(ids) >= self.max_positions:
            return None
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
    num_questions, valid_questions, not_in_dict, processing_problem = 0, 0, 0, 0
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

        entity_name = entity_name.replace(' ', '_')

        # assert entity_name in entity_dict
        if not entity_name in entity_dict:
            not_in_dict += 1
            continue
        entity_id = entity_dict.index(entity_name)

        answer_entities.append(entity_id)

        ids_tensor, processed_annotation = processor.process(question, annotation)
        processed_annotations.append(processed_annotation)
        if ids_tensor is None:
            processing_problem += 1
            continue
        question_builder.add_item(ids_tensor)
        valid_questions += 1

    pbar.close()

    question_builder.finalize(os.path.join(output_dir, split+ '.questions_entities' + '.idx'))
    np.save(os.path.join(output_dir, split+'.answer_entities'), answer_entities)

    processed_annotation_path = os.path.join(args.root_dir, args.split+'_processed_annotations.json')
    with codecs.open(processed_annotation_path, 'w', 'utf8') as f:
        json.dump(processed_annotations, f, indent=4)
    a = 3


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
