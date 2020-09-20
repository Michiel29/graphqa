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

    def strip_empty_words(self, words):
        new_words = []

        for word_id, word in enumerate(words):
            if len(word.strip()) != 0:
                new_words.append(word)
        return new_words

    def apply_gt2_bpe(self, sentence):
        global bpe
        ids = list(map(str, bpe.encode(sentence)))
        return ids

    def process(self, tokens):
        global vocab
        words = self.strip_empty_words(tokens)
        ids = self.apply_gt2_bpe(' '.join(words))

        if len(ids) >= self.max_positions:
            return None
        ids_tensor = vocab.encode_line(line=' '.join(ids), append_eos=self.append_eos)

        assert len(ids_tensor) == len(ids) + int(self.append_eos)
        return ids_tensor


def main(args):
    data_path = os.path.join(args.root_dir, args.split+'.json')
    with codecs.open(data_path, 'r', 'utf8') as f:
        data = json.load(f)["Data"]
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
        os.path.join(output_dir, split + '_entities' + '.text.bin'),
        impl=args.dataset_impl,
        vocab_size=len(vocab),
    )
    answer_entities = []

    processor.initializer()
    num_questions, valid_questions, not_in_dict, processing_problem = 0, 0, 0, 0
    for sample in data:
        num_questions += 1
        question = [sample["Question"]]
        answer = sample["Answer"]
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

        ids_tensor = processor.process(question)
        if ids_tensor is None:
            processing_problem += 1
            continue
        question_builder.add_item(ids_tensor)
        valid_questions += 1

        pbar.update()
    pbar.close()

    question_builder.finalize(os.path.join(output_dir, split+'.questions.idx'))
    np.save(os.path.join(output_dir, split+'.answer_entities'), answer_entities)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation for TriviaQA dataset')
    parser.add_argument('--split', type=str, help='Dataset split', choices=['train', 'dev', 'test'])
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
