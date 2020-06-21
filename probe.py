#!/usr/bin/env python3 -u

import os
import logging
import sys
from collections import defaultdict

import numpy as np
import torch

from fairseq import options, progress_bar, utils, tasks

from utils.config import update_namespace, modify_factory, compose_configs, update_config, save_config
from utils.dictionary import CustomDictionary, EntityDictionary
from utils.diagnostic_utils import Diagnostic
from utils.probing_utils import save_probing_results

import models
import tasks as custom_tasks

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')

NEPTUNE_PROJECT_NAME = 'selfinference/sandbox'
is_neptune_initialized = False


def main(args):
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task = tasks.setup_task(args)
    # Load valid datasets (we load training data below, based on the latest checkpoint)
    valid_subset = args.valid_subset.split(',')[0]

    task.load_dataset(valid_subset, combine=False, epoch=0)

    # Build models
    model = task.build_model(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        device = torch.device('cuda')

    # Load model checkpoint
    state_dict = torch.load(args.restore_file)['model']
    prefix = 'model_dict.gnn.'
    adjusted_state_dict = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
    model.load_state_dict(adjusted_state_dict)

    # Move model to GPU
    if use_cuda:
        model.cuda()

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

    # Load dictionary
    dict_path = os.path.join(args.data_path, 'dict.txt')
    dictionary = CustomDictionary.load(dict_path)

    log_outputs = []
    all_results, rule_results = [], defaultdict(list)
    diag = Diagnostic(dictionary, entity_dictionary=None)
    for i, sample in enumerate(progress):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        batch_size = sample['size']
        scores, target_relation, evidence_relations, decoded_rules, log_output = task.probe_step(sample, model, diag)
        for j in range(batch_size):
            all_results.append({
                'rule': tuple([target_relation[j], evidence_relations[j][0], evidence_relations[j][1]]),
                'scores': scores[j].cpu().numpy(),
                'mean_score': scores[j].mean().item(),
                'n_samples': len(scores[j]),
                'decoded_rules': decoded_rules[j]
            })
            rule_results[tuple(evidence_relations[j])].append({
                'target_relation': target_relation[j],
                'scores': scores[j].cpu().numpy(),
                'mean_score': scores[j].mean().item(),
                'n_samples': len(scores[j]),
                'decoded_rules': decoded_rules[j]
            })
        progress.log(log_output, step=i)
        log_outputs.append(log_output)

    # Sort results by mean_score
    all_results = sorted(all_results, key=lambda k: k['mean_score'], reverse=True)
    for e in rule_results.keys():
        rule_results[e] = sorted(rule_results[e], key=lambda k: k['mean_score'], reverse=True)
    
    # Save probing results
    save_dir = os.path.join(args.data_path, 'probing')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_probing_results(all_results, os.path.join(save_dir, 'all_results.pkl'))
    save_probing_results(rule_results, os.path.join(save_dir, 'rule_results.pkl'))
    save_probing_results(task.datasets['valid'].strong_neg_rules, os.path.join(save_dir, 'strong_neg_rules.pkl'))


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument('--config', type=str, nargs='*', help='paths to JSON files of experiment configurations, from high to low priority')
    parser.add_argument('--load-checkpoint', type=str, help='path to checkpoint to load (possibly composite) model from')
    parser.add_argument('--exp-name', type=str, default='', help='name of the experiment')
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='run training in the debugging mode',
    )

    pre_parsed_args = parser.parse_args()

    config_dict = {}
    for config_path in pre_parsed_args.config:
        config_dict = update_config(config_dict, compose_configs(config_path))

    parser_modifier = modify_factory(config_dict)

    args = options.parse_args_and_arch(parser, modify_parser=parser_modifier)

    update_namespace(args, config_dict)

    main(args)


if __name__ == '__main__':
    os.environ['NEPTUNE_API_TOKEN'] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMGVlMjhiMTQtZGU0YS00MDFiLWE2NzQtNDk4Y2M1NTQwY2Q4In0="
    cli_main()
