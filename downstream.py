import logging
import math
import random
import os
import sys
import argparse
import copy
from timeit import default_timer as timer

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from fairseq import (
    checkpoint_utils, criterions, distributed_utils, metrics, options, progress_bar, tasks, utils
)
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.models import ARCH_MODEL_REGISTRY

import models, criterions
import tasks as custom_tasks

from utils.config import update_namespace, modify_factory, compose_configs, update_config, save_config
from utils.checkpoint_utils import generate_save_dir

from utils.logging_utils import compute_sklearn_stats, maybe_wrap_neptune_logging
from utils.downstream_utils import (
    load_downstream_data,
    prepare_sample,
    valid_step,
    get_ft_train_stats,
    get_ft_valid_stats,
    get_sklearn_stats
)


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')


def downstream_train_pytorch(args, trainer, task, epoch_itr, train_prefix, pct, global_epoch, valid_subset):
    """Fine-tune PyTorch classifier on downstream training set for one epoch"""
    task.split = 'train'
    train_prefix = '_'.join([train_prefix, 'pct{:03d}'.format(pct)])

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
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    # Add global epoch to beginning of progress bar description
    try:
        progress.wrapped_bar.tqdm.set_description(desc='epoch {:03d} | \'{}\' {}'.format(global_epoch, train_prefix, progress.wrapped_bar.prefix), refresh=True)
    except:
        progress.tqdm.set_description(desc='epoch {:03d} | \'{}\' {}'.format(global_epoch, train_prefix, progress.tqdm.desc), refresh=True)

    progress = maybe_wrap_neptune_logging(progress, args)

    # Task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    max_update = args.max_update or math.inf
    with metrics.aggregate() as agg:
        for samples in progress:

            # Train for one step
            log_output = trainer.train_step(samples)
            num_updates = trainer.get_num_updates()
            if log_output is None:
                continue

            # Log mid-epoch stats
            stats = get_ft_train_stats(agg.get_smoothed_values())
            progress.log(stats, tag=train_prefix, step=num_updates)

            if num_updates >= max_update:
                break

    # Log end-of-epoch stats
    stats = get_ft_train_stats(agg.get_smoothed_values())
    progress.print(stats, tag=train_prefix, step=num_updates)

    # Reset epoch-level meters
    metrics.reset_meters(train_prefix)

def downstream_validate_pytorch(args, task, model, criterion, epoch_for_logging, subsets, valid_name, num_updates, global_epoch=None):
    """Evaluate the model on the validation set(s) and return the losses."""
    task.split = 'valid'
    valid_name_ = valid_name if valid_name is not None else 'valid'

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
                model.max_positions(),
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

        # Add global epoch to beginning of progress bar description
        if global_epoch is not None:
            try:
                progress.wrapped_bar.tqdm.set_description(desc='epoch {:03d} | \'{}\' {}'.format(global_epoch, valid_name_, progress.wrapped_bar.prefix), refresh=True)
            except:
                progress.tqdm.set_description(desc='epoch {:03d} | \'{}\' {}'.format(global_epoch, valid_name_, progress.tqdm.desc), refresh=True)

        progress = maybe_wrap_neptune_logging(progress, args)

        # Reset validation meters
        metrics.reset_meters(valid_name_)

        with metrics.aggregate(valid_name) as agg:
            dummy_batch = "DUMMY"
            for sample in progress:
                dummy_batch = sample if dummy_batch == "DUMMY" else dummy_batch
                valid_step(args, sample, task, model, criterion, dummy_batch, logger)

        # Log validation stats
        stats = get_ft_valid_stats(args, agg.get_smoothed_values(), num_updates)
        progress.print(stats, tag=valid_name_, step=num_updates)

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses

def downstream_train_sklearn(args, task, model, epoch_for_logging, task_name, num_updates):
    """Fine-tune sklearn classifier on downstream training set"""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=task.dataset('train'),
        max_tokens=args.max_tokens_sklearn,
        max_sentences=args.max_sentences_sklearn,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=1,
        shard_id=0,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_for_logging,
        prefix='sklearn fine-tune on \'{}\''.format(task_name),
        no_progress_bar='simple'
    )
    progress = maybe_wrap_neptune_logging(progress, args)

    # Reset meters
    metrics.reset_meters(task_name)

    # Set model to eval mode
    with torch.no_grad():
        model.eval()

    # Load downstream train data
    features, targets = load_downstream_data(args, progress, model, args.scaler)

    # Train classifier
    logger.info('fine-tuning LogisticRegression classifier on \'{}\''.format(task_name))
    timer_start = timer()
    if args.cross_validation:
        classifier = LogisticRegressionCV(
            multi_class=args.multi_class,
            solver=args.solver,
            Cs=args.Cs,
            cv=args.k_folds,
            n_jobs=min(os.cpu_count(), args.num_classes, args.Cs*args.k_folds),
            random_state=args.seed,
            max_iter=args.max_iter,
            verbose=args.verbose
        ).fit(features, targets)
    else:
        classifier = LogisticRegression(
            multi_class=args.multi_class,
            solver=args.solver,
            n_jobs=min(os.cpu_count(), args.num_classes, args.n_jobs),
            random_state=args.seed,
            max_iter=args.max_iter,
            verbose=args.verbose
        ).fit(features, targets)
    timer_end = timer()
    logger.info('finished sklearn fine-tuning in {:.2f} seconds'.format(timer_end-timer_start))

    # Compute class predictions and probabilities
    class_predictions = classifier.predict(features)
    class_probabilities = classifier.predict_proba(features)

    # Compute and log downstream training stats
    stats = compute_sklearn_stats(targets, class_predictions, class_probabilities, args.num_classes, args.eval_metric)
    stats = get_sklearn_stats(stats, num_updates)
    progress.print(stats, tag=task_name+'_sk_train', step=num_updates)

    return classifier

def downstream_validate_sklearn(args, task, model, epoch_for_logging, task_name, num_updates, classifier):
    """Evaluate classifier on downstream validation set"""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=task.dataset('valid'),
        max_tokens=args.max_tokens_sklearn,
        max_sentences=args.max_sentences_sklearn,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=1,
        shard_id=0,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_for_logging,
        prefix='sklearn valid on \'{}\''.format(task_name),
        no_progress_bar='simple'
    )
    progress = maybe_wrap_neptune_logging(progress, args)

    # Reset validation meters
    metrics.reset_meters(task_name)

    # Set model to eval mode
    with torch.no_grad():
        model.eval()

    # Load downstream validation data
    features, targets = load_downstream_data(args, progress, model, args.scaler)

    # Compute class predictions and probabilities
    class_predictions = classifier.predict(features)
    class_probabilities = classifier.predict_proba(features)

    # Compute and log downstream validation stats
    stats = compute_sklearn_stats(targets, class_predictions, class_probabilities, args.num_classes, args.eval_metric)
    stats = get_sklearn_stats(stats, num_updates)
    progress.print(stats, tag=task_name+'_sk_valid', step=num_updates)