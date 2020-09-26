import argparse
import json
import codecs
import os
import numpy as np

def main(args):
    data_path = os.path.join(args.root_dir, 'train.json')
    with codecs.open(data_path, 'r', 'utf8') as f:
        full_dataset = json.load(f)

    data = full_dataset["Data"]
    np.random.seed(0)
    small_indices = np.random.permutation(args.size)
    small_data = [data[i] for i in small_indices]

    small_dataset = {key:value for key, value in full_dataset.items()}
    small_dataset["Data"] = small_data

    output_path = os.path.join(args.root_dir, 'train_' + str(args.size) + '.json')
    with codecs.open(output_path, 'w', 'utf8') as f:
        json.dump(small_dataset, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select subset of TriviaQA dataset')
    parser.add_argument('--root-dir', type=str, default='../data/triviaqa/triviaqa/', help='TriviaQA root directory')
    parser.add_argument('--size', type=int, default=10000, help='TriviaQA root directory')
    args = parser.parse_args()
    main(args)