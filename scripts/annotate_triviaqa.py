import argparse
import json
import codecs
import os
from tqdm import tqdm
import time

# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

entity_type_names=['UNKNOWN', 'PERSON', 'LOCATION', 'ORGANIZATION', 'EVENT', 'WORK_OF_ART', 'CONSUMER_GOOD', 'OTHER',
'OTHER', 'NONE', 'PHONE_NUMBER', 'ADDRESS', 'DATE', 'NUMBER', 'PRICE']
mention_type_names=['UNKNOWN', 'PROPER', 'COMMON']

def convert_to_dictionary(entity):

    new_dict = {}

    new_dict['name'] = entity.name
    new_dict['salience'] = entity.salience

    new_mentions = []
    for mention in entity.mentions:
        new_mention = {
            'type': mention_type_names[mention.type],
            'content': mention.text.content,
            'begin_offset': mention.text.begin_offset
        }
        new_mentions.append(new_mention)

    new_dict['mentions'] = new_mentions

    new_dict['type'] = entity_type_names[entity.type]
    metadata = getattr(entity, 'metadata', None)
    new_dict['metadata'] = dict(metadata) if metadata else None

    return new_dict


def main(args):
    # Instantiates a client
    client = language.LanguageServiceClient()

    data_path = os.path.join(args.root_dir, args.split+'.json')
    with codecs.open(data_path, 'r', 'utf8') as f:
        data = json.load(f)["Data"]
    annotations = list()

    total, failed = 0, 0
    for data_point in tqdm(data):
        total += 1
        question = data_point['Question']
        document = types.Document(
        content=question,
        type=enums.Document.Type.PLAIN_TEXT)
        time.sleep(0.08)

        try:
            entities = client.analyze_entities(document=document).entities
            question_annotations = list()
            for entity in entities:
                entity_dictionary = convert_to_dictionary(entity)
                question_annotations.append(entity_dictionary)
            annotations.append(question_annotations)
        except Exception as e:
            annotations.append(None)
            failed += 1

        if total % 1000 == 0 and total != 0:
            print('Failed annotations: ' + str(failed))


    annotation_path = os.path.join(args.root_dir, args.split+'_annotations.json')
    with codecs.open(annotation_path, 'w', 'utf8') as f:
        json.dump(annotations, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entity annotation for TriviaQA dataset')
    parser.add_argument('--split', type=str, help='Dataset split, usually train, dev or test', default='train_10000')
    parser.add_argument('--root-dir', type=str, default='../data/triviaqa/triviaqa/', help='TriviaQA root directory')
    args = parser.parse_args()
    main(args)