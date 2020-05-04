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


def main(args, init_distributed=False):
    utils.import_user_module(args)

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
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

    # Load valid datasets (we load training data below, based on the latest checkpoint)
    valid_subsets = args.valid_subset.split(',')
    for valid_sub_split in valid_subsets:
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build models
    model = task.build_model(args)

    # Build criterions
    criterion = task.build_criterion(args)

    logger.info(model)
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    logger.info('training on {} GPUs'.format(args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    if distributed_utils.is_master(args) and not args.debug:
        initialize_neptune(trainer, extra_state, args)

    if args.eval_downstream and len(args.downstream_dict) > 0:
        downstream_dict = {}
        for downstream_name, downstream_kwargs in args.downstream_dict.items():
            downstream_dict[downstream_name] = create_downstream_dict(args, downstream_name, downstream_kwargs, model)

    # Move model and criterion to gpu
    if torch.cuda.is_available() and not args.cpu:
        model.to('cuda:{}'.format(args.device_id))
        criterion.to('cuda:{}'.format(args.device_id))

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf

    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    if args.validate_before_training and extra_state is None:
        # We want to make sure we do validate_before_training
        # only when we start the trainig from scratch (thus, extra_state is None).
        # Here, we assert that indeed the training has just started
        # and training epoch is equal to one.
        assert epoch_itr.epoch == 1
        valid_losses = validate(args, trainer, task, 0, valid_subsets)
        if args.eval_downstream:
            run_downstream(args, downstream_dict, model, criterion, 0, trainer.get_num_updates())

    while (
        not args.disable_training
        and (
            (isinstance(lr, np.ndarray) and all(lr > args.min_lr))
            or (not isinstance(lr, np.ndarray) and lr > args.min_lr)
        )
        and epoch_itr.next_epoch_idx <= max_epoch
        and trainer.get_num_updates() < max_update
    ):
        train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:

            # validate on task validation set
            valid_losses = validate(args, trainer, task, epoch_itr.epoch, valid_subsets)

            # evaluate on downstream tasks
            if args.eval_downstream:
                run_downstream(args, downstream_dict, model, criterion, epoch_itr.epoch, trainer.get_num_updates())
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:

            save_extra_state = {}
            if get_experiment_id():
                save_extra_state['neptune_id'] = get_experiment_id()
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0], save_extra_state)

        # early stop
        if should_stop_early(args, valid_losses[0]):
            logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
            break

        reload_dataset = getattr(args, 'reload', False)
        # sharded data: get train iterator for next epoch
        epoch_itr = trainer.get_train_iterator(epoch_itr.next_epoch_idx, load_dataset=reload_dataset)

    train_meter.stop()
    logger.info('done training in {:.1f} seconds'.format(train_meter.sum))


def should_stop_early(args, valid_loss):
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        return should_stop_early.num_runs > args.patience


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    task.split = 'train'

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = maybe_wrap_neptune_logging(
        progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch, no_progress_bar='simple',
        ),
        args=args,
    )

    # task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    with metrics.aggregate() as agg:
        for samples in progress:
            log_output = trainer.train_step(samples)
            num_updates = trainer.get_num_updates()
            if log_output is None:
                continue

            # log mid-epoch stats
            stats = get_training_stats(agg.get_smoothed_values())
            progress.log(stats, tag='train', step=num_updates)

            if num_updates >= max_update:
                break

    # log end-of-epoch stats
    stats = get_training_stats(agg.get_smoothed_values())
    progress.print(stats, tag='train', step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters('train')


def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_for_logging, subsets, valid_name=None, num_updates=None, global_epoch=None):
    """Evaluate the model on the validation set(s) and return the losses."""
    task.split = 'valid'
    valid_name_ = valid_name if valid_name is not None else 'valid'
    num_updates = trainer.get_num_updates() if num_updates is None else num_updates

    if args.fixed_validation_seed is not None:
        # Set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
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
            epoch=1,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_for_logging,
            prefix='valid on \'{}\' subset'.format(valid_name_),
            no_progress_bar='simple'
        )
        progress = maybe_wrap_neptune_logging(progress, args)

        # Add global epoch to beginning of progress bar description
        if global_epoch is not None:
            try:
                progress.wrapped_bar.tqdm.set_description(desc='epoch {:03d} | \'{}\' {}'.format(global_epoch, valid_name_, progress.wrapped_bar.prefix), refresh=True)
            except:
                progress.tqdm.set_description(desc='epoch {:03d} | \'{}\' {}'.format(global_epoch, valid_name_, progress.tqdm.desc), refresh=True)

        # Reset validation meters
        metrics.reset_meters(valid_name_)

        with metrics.aggregate(valid_name) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # Log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=valid_name_, step=num_updates)

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


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


def run_downstream(args, downstream_dict, model, criterion, global_epoch, num_updates):
    for downstream_name in downstream_dict:

        downstream_args = downstream_dict[downstream_name]['args']
        downstream_task = downstream_dict[downstream_name]['task']

        # Do sklearn fine-tuning
        if (
                downstream_args.task_type == 'supervised'
                and (
                        global_epoch % (downstream_args.epoch_interval or 1) == 0
                        or global_epoch == downstream_args.max_epoch
                    )
                and args.distributed_rank == 0
            ):
                # Fine-tune sklearn LogisticRegression classifier on downstream_task validation set
                sk_classifier, sk_scaler = downstream_train_sklearn(
                    downstream_args,
                    downstream_task,
                    model,
                    global_epoch,
                    downstream_name,
                    num_updates
                )

                # Evaluate sklearn LogisticRegression classifier on downstream_task validation set
                downstream_validate_sklearn(
                    downstream_args,
                    downstream_task,
                    model,
                    global_epoch,
                    downstream_name,
                    num_updates,
                    sk_classifier,
                    sk_scaler
                )

        # Evaluate few-shot classifier
        elif (
            downstream_args.task_type == 'few_shot'
            and (
                    global_epoch % (downstream_args.epoch_interval or 1) == 0
                    or global_epoch == downstream_args.max_epoch
                )
        ):
            downstream_model = downstream_dict[downstream_name]['model']
            downstream_criterion = downstream_dict[downstream_name]['criterion']
            downstream_valid_subset = downstream_dict[downstream_name]['valid_subset']

            downstream_validate_pytorch(
                downstream_args,
                downstream_task,
                downstream_model,
                downstream_criterion,
                global_epoch,
                downstream_valid_subset,
                downstream_name,
                num_updates
            )


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

    training_name = get_training_name(args)
    base_save_dir = generate_save_dir(args, training_name, sys.argv[1:])
    setattr(args, 'training_name', training_name)
    setattr(args, 'save_dir', os.path.join(base_save_dir, 'checkpoints'))
    setattr(args, 'tensorboard_logdir', os.path.join(base_save_dir, 'tensorboard'))

    save_config(vars(args), base_save_dir)

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
