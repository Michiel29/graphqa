import logging
import random
import os
import sys
import argparse
import copy

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq import (
    checkpoint_utils, criterions, distributed_utils, metrics, options, progress_bar, tasks, utils
)
from fairseq.data import iterators

import models, criterions
import tasks as custom_tasks
from trainer import Trainer

from utils.config import update_namespace, modify_factory, compose_configs, update_config, save_config
from utils.diagnostic_utils import Diagnostic

def main(args):

    task = tasks.setup_task(args)
    # Load valid datasets (we load training data below, based on the latest checkpoint)
    valid_subset = args.valid_subset.split(',')[0]

    task.load_dataset(valid_subset, combine=False, epoch=0)

    # Build models
    model = task.build_model(args)

    cuda = torch.cuda.is_available() and not args.cpu
    if cuda:
        device = torch.device('cuda')

    # Load model checkpoint
    state_dict = torch.load(args.restore_file)['model']
    prefix = 'model_dict.gnn.'
    adjusted_state_dict = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
    model.load_state_dict(adjusted_state_dict)

    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=task.dataset(valid_subset),
        max_tokens=args.max_tokens_valid,
        max_sentences=args.max_sentences_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_workers=args.num_workers,
        epoch=1,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args, itr, 0,
        prefix='eval gnn',
        no_progress_bar='simple'
    )

    ent_dictionary = task.entity_dictionary
    all_scores = []
    targets = []
    supporting = []
    entities = {'A': [], 'B': [], 'C': []}

    for sample in progress:
        text = torch.cat(sample['text'], dim=0)
        targets.append(text[sample['target_text_idx']])
        supporting.append(text[sample['graph']])
        scores = model(sample)
        all_scores.append(scores)
        for entity in entities:
            entities[entity].append(sample['entities'][entity])

    all_scores = torch.cat(all_scores, dim=0).cpu().detach().numpy()
    targets = torch.cat(targets, dim=0).cpu().detach().numpy()
    supporting = torch.cat(supporting, dim=0).cpu().detach().numpy()
    entities = {key: torch.cat(value, dim=0).cpu().detach().numpy() for key, value in entities.items()}
    sort_idx = np.argsort(all_scores)[::-1]
    sorted_scores = all_scores[sort_idx]
    sorted_targets = targets[sort_idx]
    sorted_supporting = supporting[sort_idx]
    entities = {entity: np.array(value) for entity, value in entities.items()}
    sorted_entities = {entity: value[sort_idx] for entity, value in entities.items()}
    diag = Diagnostic(task.dictionary, task.entity_dictionary)

    n_random_examples = 100
    n_top_examples = 100

    with open('/tmp/random_examples', 'w') as f:
        for i in np.random.randint(len(targets), size=n_random_examples):
            a_b_text = diag.decode_text(targets[i]).replace('<s_head> <blank> <e_head>', '<<<A>>>').replace('<s_tail> <blank> <e_tail>', '<<<B>>>')
            a_c_text = diag.decode_text(supporting[i, 0]).replace('<s_head> <blank> <e_head>', '<<<A>>>').replace('<s_tail> <blank> <e_tail>', '<<<C>>>')
            c_b_text = diag.decode_text(supporting[i, 1]).replace('<s_head> <blank> <e_head>', '<<<C>>>').replace('<s_tail> <blank> <e_tail>', '<<<B>>>')
            f.write('A: %s, B: %s, C: %s\nScore:\t%s\nA_B:\t%s\nA_C:\t%s\nC_B:\t%s\n\n' % (
            ent_dictionary[entities['A'][i]], ent_dictionary[entities['B'][i]], ent_dictionary[entities['C'][i]], all_scores[i], a_b_text, a_c_text, c_b_text))

    with open('/tmp/top_examples', 'w') as f:
        for i in range(n_top_examples):
            a_b_text = diag.decode_text(sorted_targets[i]).replace('<s_head> <blank> <e_head>', '<<<A>>>').replace('<s_tail> <blank> <e_tail>', '<<<B>>>')
            a_c_text = diag.decode_text(sorted_supporting[i, 0]).replace('<s_head> <blank> <e_head>', '<<<A>>>').replace('<s_tail> <blank> <e_tail>', '<<<C>>>')
            c_b_text = diag.decode_text(sorted_supporting[i, 1]).replace('<s_head> <blank> <e_head>', '<<<C>>>').replace('<s_tail> <blank> <e_tail>', '<<<B>>>')
            f.write('A: %s, B: %s, C: %s\nScore:\t%s\nA_B:\t%s\nA_C:\t%s\nC_B:\t%s\n\n' % (ent_dictionary[sorted_entities['A'][i]], ent_dictionary[sorted_entities['B'][i]], ent_dictionary[sorted_entities['C'][i]], sorted_scores[i], a_b_text, a_c_text, c_b_text))

    print('done')

if __name__ == '__main__':

    parser = options.get_training_parser()
    parser.add_argument(
        '--config',
        type=str,
        nargs='*',
        help='paths to JSON files of experiment configurations, from high to low priority',
    )
    parser.add_argument('--torch-file-system', action='store_true')
    pre_parsed_args, unknown = parser.parse_known_args()

    config_dict = {}
    for config_path in pre_parsed_args.config:
        config_dict = update_config(config_dict, compose_configs(config_path))

    parser_modifier = modify_factory(config_dict)

    args = options.parse_args_and_arch(parser, modify_parser=parser_modifier)

    update_namespace(args, config_dict)

    # set sharing strategy file system in case /dev/shm/ limits are small
    if args.torch_file_system:
        torch.multiprocessing.set_sharing_strategy('file_system')

    main(args)


