#!/usr/bin/env python3 -u

import os
import logging
import sys
from collections import defaultdict

import numpy as np
import torch

from fairseq import checkpoint_utils, metrics, options, progress_bar, utils, tasks

from utils.config import update_namespace, modify_factory, compose_configs, update_config, save_config
from utils.checkpoint_utils import select_component_state, handle_state_dict_keys
from utils.dictionary import CustomDictionary, EntityDictionary
from utils.diagnostic_utils import Diagnostic

import models, criterions
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
    utils.import_user_module(args)
    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)
    task.split = 'valid'

    # Load model
    load_checkpoint = getattr(args, 'load_checkpoint')

    if load_checkpoint:
        logger.info('loading model(s) from {}'.format(load_checkpoint))
        if not os.path.exists(load_checkpoint):
            raise IOError("Model file not found: {}".format(load_checkpoint))
        state = checkpoint_utils.load_checkpoint_to_cpu(load_checkpoint)

        checkpoint_args = state["args"]
        if task is None:
            task = tasks.setup_task(args)

        load_component_prefix = getattr(args, 'load_component_prefix', None)

        model_state = state["model"]
        if load_component_prefix:
            model_state = select_component_state(model_state, load_component_prefix)

        # build model for ensemble
        model = task.build_model(args)
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False, args=args)
        handle_state_dict_keys(missing_keys, unexpected_keys)

    else:
        model = task.build_model(args)

    # Move model to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(args)

    # Build criterion
    criterion = task.build_criterion(args)
    criterion.eval()

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for subset in args.valid_subset.split(','):
        try:
            task.load_dataset(subset, combine=False, epoch=0)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception('Cannot find dataset: ' + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                model.max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr,
            prefix='valid on \'{}\' subset'.format(subset),
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
            scores, target_relation, evidence_relations, decoded_rules, log_output = task.probe_step(sample, model, diag)
            all_results.append({
                'rule': tuple([target_relation, evidence_relations[0], evidence_relations[1]]),
                'scores': scores,
                'mean_score': scores.mean().item(),
                'n_samples': len(scores),
                'decoded_rules': decoded_rules
            })
            rule_results[evidence_relations].append({
                'target_relation': target_relation,
                'scores': scores,
                'mean_score': scores.mean().item(),
                'n_samples': len(scores),
                'decoded_rules': decoded_rules
            })
            progress.log(log_output, step=i)
            log_outputs.append(log_output)

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        progress.print(log_output, tag=subset, step=i)

        # Sort results by mean_score
        all_results = sorted(all_results, key=lambda k: k['mean_score'], reverse=True)
        for e in rule_results.keys():
            rule_results[e] = sorted(rule_results[e], key=lambda k: k['mean_score'], reverse=True)
        

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
