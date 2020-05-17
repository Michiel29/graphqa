#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import math
import random
import os
import sys
import argparse
import copy
import gc
import glob

import numpy as np
import torch

from fairseq import (
    checkpoint_utils, criterions, distributed_utils, metrics, options, progress_bar, tasks, utils
)
from fairseq.data import iterators
from fairseq.meters import StopwatchMeter

import models, criterions
import tasks as custom_tasks
from trainer import Trainer

from downstream import (
    downstream_train_pytorch,
    downstream_validate_pytorch,
    downstream_train_sklearn,
    downstream_validate_sklearn
)
from utils.config import update_namespace, modify_factory, compose_configs, update_config, save_config
from utils.checkpoint_utils import (
    generate_save_dir,
    get_training_name,
    save_checkpoint,
)
from utils.downstream_utils import (
    create_ft_prefixes,
    create_downstream_dict,
    setup_ft_args,
    setup_ft_tasks,
    setup_ft_model,
    setup_ft_criterion
)
from utils.logging_utils import (
    compute_sklearn_stats,
    maybe_wrap_neptune_logging,
    NeptuneWrapper,
    initialize_neptune,
    get_experiment_id
)

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')


NEPTUNE_PROJECT_NAME = 'selfinference/sandbox'
is_neptune_initialized = False

# Create param prefix dict
param_prefix_dict = {
    'pct_train_examples': 'pct',
    'n_train_relations': 'rel',
    'n_train_examples_per_relation': 'epr'
}


def main(args, init_distributed=False):
    utils.import_user_module(args)

    # Initialize CUDA and distributed training
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup tasks, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Get base path
    training_name = get_training_name(args)
    base_path = os.path.join(args.save_dir, training_name)

    # Iterate through each item in ckpt dict
    for ckpt_id, ckpt_item in args.checkpoint_dict.items():

        # Set up trainer
        trainer = Trainer(args, task, model, criterion)

        # Intialize neptune experiment
        setattr(args, 'training_name', os.path.join(training_name, 'ft', ckpt_id.split('/')[-1]))
        if distributed_utils.is_master(args) and not args.debug:
            initialize_neptune(trainer, None, args)

        # Create list of ckpt paths for current ckpt_item
        ckpt_base_path = os.path.join(base_path, ckpt_id)
        if 'checkpoints' not in ckpt_item.keys():
            ckpt_list = list(glob.glob(os.path.join(ckpt_base_path, 'checkpoints/*.pt'), recursive=True))
        elif len(ckpt_item['checkpoints']) == 0:
            ckpt_list = list(glob.glob(os.path.join(ckpt_base_path, 'checkpoints/*.pt'), recursive=True))
        else:
            ckpt_list = [os.path.join(ckpt_base_path, 'checkpoints', x) for x in ckpt_item['checkpoints']]

        # Filter checkpoint_best and checkpoint_last out of ckpt_list
        ckpt_list = [x for x in ckpt_list if x.split('/')[-1][-7:-3] != 'best' and x.split('/')[-1][-7:-3] != 'last']

        # Iterate through each ckpt path for current ckpt_item
        for ckpt in ckpt_list:
            ckpt_idx = int(ckpt[-4])
            evaluate_checkpoint(args, ckpt, ckpt_idx, trainer)


def evaluate_checkpoint(args, ckpt, ckpt_idx, trainer):
    use_cuda = torch.cuda.is_available() and not args.cpu

    # Set checkpoint path in args
    setattr(args, 'restore_file', ckpt)

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    checkpoint_utils.load_checkpoint(args, trainer)

    # Create ft dict
    ft_dict = {}
    for ft_name, ft_kwargs in args.downstream_dict.items():
        # Set random seed
        np.random.seed(args.seed); torch.manual_seed(args.seed) 

        # Get task_dict for current task
        ft_dict[ft_name] = create_downstream_dict(args, ft_name, ft_kwargs, trainer.get_model()) 

    # Fine-tune model on each downstream task    
    for ft_name in ft_dict:
        # Set random seed
        np.random.seed(args.seed); torch.manual_seed(args.seed)

        # Set up param_types list
        if ft_name in ['semeval2010task8', 'kbp37', 'tacred']:
            param_types = ['pct_train_examples']
        elif ft_name in ['fewrel_0', 'fewrel_1']:
            param_types = ['n_train_relations', 'n_train_examples_per_relation']
        else:
            raise NotImplementedError
        
        # Run fine-tuning for current task
        run_ft(args, ft_dict, ft_name, param_types, ckpt_idx)


def run_ft(args, ft_dict, ft_name, param_types, ckpt_idx):
    ft_task_dict = ft_dict[ft_name]
    ft_args_orig = ft_task_dict['args']
    ft_task_orig = ft_task_dict['task']
    ft_valid_subsets = ft_task_dict['valid_subset']
    
    logger.info('training on {} GPUs'.format(args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        ft_args_orig.max_tokens,
        ft_args_orig.max_sentences,
    ))

    # Iterate through each param type
    for param_type in param_types:
        # Set random seed
        np.random.seed(args.seed); torch.manual_seed(args.seed)

        # Set up param list
        if param_type == 'pct_train_examples':
            params = sorted(list(set(ft_args_orig.pct_train_examples)))
            params = ft_args_orig.pct_train_examples
            assert all([x > 0 and x <= 100 for x in params]) # make sure pct_train_examples vals are inside (0, 100]
        elif param_type == 'n_train_relations':
            params = sorted(list(set(ft_args_orig.n_train_relations)))
            assert all([x >= 5 and x <= 64 for x in params]) # make sure n_train_relations vals are within [5, 64]
        elif param_type == 'n_train_examples_per_relation':
            params = sorted(list(set(ft_args_orig.n_train_examples_per_relation)))
            assert all([x >= 5 and x <= 700 for x in params]) # make sure n_train_examples_per_relation vals are within [5, 700]

        # Set up ft_args_list and ft_task_list for each param value
        ft_args_list = setup_ft_args(params, param_type, ft_args_orig, ft_task_orig)
        ft_task_list = setup_ft_tasks(ft_args_list, ft_valid_subsets)
    
        # Iterate through params
        for param_idx, param in enumerate(params):
            # Set random seed
            np.random.seed(args.seed); torch.manual_seed(args.seed)

            # Get ft_args and ft_task for current param
            ft_args = ft_args_list[param_idx]
            ft_task = ft_task_list[param_idx]
            
            # Train and validate model for given task and param
            ft_train_validate(ft_args, ft_task, ft_task_dict, param, param_type, ckpt_idx)


def ft_train_validate(ft_args, ft_task, ft_task_dict, param, param_type, ckpt_idx):
    use_cuda = torch.cuda.is_available() and not ft_args.cpu
    ft_max_epoch = ft_args.max_epoch or math.inf

    # Set up ft prefixes
    ft_train_prefix, ft_valid_prefix = create_ft_prefixes(
        ft_task_dict['name'], 
        param, 
        param_prefix_dict[param_type],
        ckpt_idx
    )

    # Get ft_model for current param
    ft_model = setup_ft_model(ft_args, ft_task_dict['model'], use_cuda)

    # Get ft_criterion for current pct
    ft_criterion = setup_ft_criterion(ft_args, ft_task_dict['criterion'], use_cuda)

    # Instantiate ft_trainer -- which includes the ft optimizer -- for current param
    ft_trainer = Trainer(ft_args, ft_task, ft_model, ft_criterion)

    # Set up epoch iterator for current param
    ft_epoch_itr = ft_trainer.get_train_iterator(1)

    # Initialize list of validation scores
    ft_valid_scores = []

    # Start training timer
    train_meter = StopwatchMeter()
    train_meter.start()

    # Start training loop
    while ft_epoch_itr.next_epoch_idx <= ft_max_epoch:

        # Fine-tune PyTorch classifier on ft_task training set
        downstream_train_pytorch(
            ft_args, 
            ft_trainer, 
            ft_task, 
            ft_epoch_itr, 
            ft_train_prefix
        )

        # Validate PyTorch classifier on ft_task validation set
        ft_valid_loss, ft_valid_score, progress = validate(
            ft_args, 
            ft_trainer, 
            ft_task, 
            ft_epoch_itr.epoch, 
            ft_valid_prefix,
            ckpt_idx
        )
        ft_valid_scores.append(ft_valid_score)

        # only use first validation loss to update the learning rate
        ft_trainer.lr_step(ft_epoch_itr.epoch, ft_valid_loss)

        # sharded data: get train iterator for next epoch
        reload_dataset = getattr(ft_args, 'reload', False)
        ft_epoch_itr = ft_trainer.get_train_iterator(ft_epoch_itr.next_epoch_idx, load_dataset=reload_dataset)
    
    progress.print({ft_args.eval_metric: max(ft_valid_scores)}, tag=ft_valid_prefix, step=ckpt_idx)

    # Stop training timer
    train_meter.stop()
    logger.info('done training {} in {:.1f} seconds'.format(ft_task_dict['name'], train_meter.sum))


def validate(args, trainer, task, epoch_for_logging, valid_name, ckpt_idx):
    """Evaluate the model on the validation set(s) and return the losses."""
    task.split = 'valid'

    if args.fixed_validation_seed is not None:
        # Set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=task.dataset('valid'),
        max_tokens=args.max_tokens_valid,
        max_sentences=args.max_sentences_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
        epoch=epoch_for_logging,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_for_logging,
        prefix='valid on \'{}\' subset'.format(valid_name),
        no_progress_bar='simple'
    )
    progress = maybe_wrap_neptune_logging(progress, args)

    # Reset validation meters
    metrics.reset_meters(valid_name)

    with metrics.aggregate(valid_name) as agg:
        for sample in progress:
            trainer.valid_step(sample)

    # Get validation stats
    stats = get_valid_stats(args, trainer, agg.get_smoothed_values())

    # Return validations score
    return stats[args.best_checkpoint_metric], stats[args.eval_metric], progress


def get_valid_stats(args, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():

    parser = options.get_training_parser()
    parser.add_argument(
        '--config',
        type=str,
        nargs='*',
        help='paths to JSON files of experiment configurations, from high to low priority',
    )
    parser.add_argument('--exp-name', type=str, default='', help='name of the experiment')
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='run training in the debugging mode',
    )
    parser.add_argument('--path-attributes', type=str, nargs='*', default=['task', 'arch', 'lr'])
    pre_parsed_args, unknown = parser.parse_known_args()

    config_dict = {}
    for config_path in pre_parsed_args.config:
        config_dict = update_config(config_dict, compose_configs(config_path))

    parser_modifier = modify_factory(config_dict)

    args = options.parse_args_and_arch(parser, modify_parser=parser_modifier)

    update_namespace(args, config_dict)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if (
            args.update_freq is not None
            and max(args.update_freq) > 1
            and args.ddp_backend != 'no_c10d'
        ):
            logger.info('NOTE: you may get faster training with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':

    cli_main()