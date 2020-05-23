import os
import sys
import copy
from collections import defaultdict
from itertools import chain
from typing import List, Dict, Any
from math import ceil

import numpy as np
import torch
from fairseq import utils, tasks, metrics, distributed_utils, checkpoint_utils
from fairseq.trainer import Trainer
from fairseq.models import ARCH_MODEL_REGISTRY

from utils.config import update_namespace, compose_configs, update_config

from sklearn.preprocessing import StandardScaler


def create_downstream_dict(args, downstream_name, downstream_kwargs, model):
    # Create downstream_args, by overwriting a copy of args with downstream args
    downstream_args = copy.deepcopy(args)
    if 'add_configs' in downstream_kwargs:
        for config_path in downstream_kwargs['add_configs']:
            downstream_kwargs = update_config(downstream_kwargs, compose_configs(os.path.join('config', config_path)))
    update_namespace(downstream_args, downstream_kwargs, override=True)

    # Set up downstream task
    downstream_task = tasks.setup_task(downstream_args)

    # Load downstream datasets
    downstream_valid_subsets = downstream_args.valid_subset.split(',')
    for split in ['train'] + downstream_valid_subsets:
        downstream_task.load_dataset(split, combine=False, epoch=1)

    # Set up downstream eval model
    encoder = model.encoder if hasattr(model, 'encoder') else model
    downstream_model = ARCH_MODEL_REGISTRY[downstream_args.arch].build_model(downstream_args, downstream_task, encoder)

    # Set up downstream eval criterion
    downstream_criterion = downstream_task.build_criterion(downstream_args)

    if torch.cuda.is_available() and not args.cpu:
        downstream_model.to('cpu')
        downstream_criterion.to('cpu')

    # Build downstream dict
    downstream_dict = {
        'name': downstream_name,
        'args': downstream_args,
        'task': downstream_task,
        'model': downstream_model,
        'criterion': downstream_criterion,
        'valid_subset': downstream_valid_subsets
    }

    return downstream_dict

def create_ft_prefixes(downstream_name, param=None, param_prefix=None, ckpt_idx=None, global_epoch=None):
    if global_epoch is not None:
        ft_train_prefix = '_'.join([downstream_name, 'ft', 'train', 'epoch{:03d}'.format(global_epoch)])
        ft_valid_prefix = '_'.join([downstream_name, 'ft', 'valid', 'epoch{:03d}'.format(global_epoch)])
    else:
        ft_train_prefix = '_'.join([downstream_name, 'ft', 'train'])
        ft_valid_prefix = '_'.join([downstream_name, 'ft', 'valid'])

    if param is not None and param_prefix is not None:
        if param_prefix == 'rel':
            ft_train_prefix = '_'.join([ft_train_prefix, '{}{:02d}'.format(param_prefix, param), 'ckpt{}'.format(ckpt_idx)])
            ft_valid_prefix = '_'.join([ft_valid_prefix, '{}{:02d}'.format(param_prefix, param)])
        else:
            ft_train_prefix = '_'.join([ft_train_prefix, '{}{:03d}'.format(param_prefix, param), 'ckpt{}'.format(ckpt_idx)])
            ft_valid_prefix = '_'.join([ft_valid_prefix, '{}{:03d}'.format(param_prefix, param)])
    
    if global_epoch is not None:
        global_ft_valid_prefix = '_'.join([downstream_name, 'ft', 'valid'])
        return ft_train_prefix, ft_valid_prefix, global_ft_valid_prefix
    else:
        return ft_train_prefix, ft_valid_prefix

def setup_ft_args(params, param_type, downstream_args, downstream_task=None):
    assert param_type in ['pct_train_examples', 'n_train_relations', 'n_train_examples_per_relation']
    # Set up ft_args -- i.e., copy of downstream_args with n_train_examples updated
    ft_args_list = []
    for param in params:
        ft_args = copy.deepcopy(downstream_args)
        if param_type == 'pct_train_examples':
            pct = param / 100
            downstream_task.datasets['train'].set_epoch(0)
            n_train_examples = round(pct * len(downstream_task.datasets['train']))
            ft_args.n_train_examples = n_train_examples
        elif param_type == 'n_train_relations':
            ft_args.n_train_relations = param
            n_train_examples = param * 700
        elif param_type == 'n_train_examples_per_relation':
            ft_args.n_train_examples_per_relation = param
            n_train_examples = param * 64
        n_updates = ceil(n_train_examples / ft_args.max_sentences)
        ft_args.warmup_updates = round(n_updates * 0.06)
        ft_args_list.append(ft_args)
    return ft_args_list
    
def setup_ft_tasks(params, param_type, ft_args_list, ft_valid_subsets):
    # Set up ft_task, using ft_args
    ft_task_list = []
    for i, ft_args in enumerate(ft_args_list):
        ft_task = tasks.setup_task(ft_args)
        if param_type in ['n_train_relations', 'n_train_examples_per_relation']:
            ft_task.load_dataset('train', combine=False, epoch=1, prune_type=param_type, prune_param=params[i])
        else:
            ft_task.load_dataset('train', combine=False, epoch=1)
        for valid_sub_split in ft_valid_subsets:
            ft_task.load_dataset(valid_sub_split, combine=False, epoch=0)
        ft_task_list.append(ft_task)
    return ft_task_list

def setup_ft_model(args, downstream_model, use_cuda):
    if use_cuda:
        downstream_model.to('cpu') # Move original model (and optimizer) to cpu
        ft_model = copy.deepcopy(downstream_model) # Make ft_model, a copy of original model
        ft_model.to('cuda:{}'.format(args.device_id)) # Move ft_model (and optimizer) to gpu
    else:
        ft_model = copy.deepcopy(downstream_model)

    return ft_model

def setup_ft_criterion(args, downstream_criterion, use_cuda):
    if use_cuda:
        downstream_criterion.to('cpu') # Move original criterion to cpu
        ft_criterion = copy.deepcopy(downstream_criterion) # Make ft_criterion, a copy of original criterion
        ft_criterion.to('cuda:{}'.format(args.device_id)) # Move ft_criterion to gpu
    else:
        ft_criterion = copy.deepcopy(downstream_criterion)
    return ft_criterion

def load_downstream_data(args, samples, model, split, scaler=None, scaler_type=None):
    assert not(scaler != None and scaler_type != None)
    assert split in ['train', 'valid']
    assert scaler_type in ['standard', None]

    features, targets = None, None
    for batch in samples:
        if batch is None:
            continue
        batch = prepare_sample(args, batch)
        text_enc, _ = model.encoder(batch['text'], annotation=batch.get('annotation'))
        batch_features = text_enc.cpu().detach().numpy()
        batch_targets = batch['target'].cpu().detach().numpy()

        if features is None:
            features, targets = batch_features, batch_targets
        else:
            features = np.concatenate((features, batch_features), axis=0)
            targets = np.concatenate((targets, batch_targets), axis=0)

    if split == 'train':
        # If there are fewer than n_splits examples for a given class, make 
        # duplicates of that class' examples, so that each split has at
        # least one example.
        unique_targets, target_counts = np.unique(targets, return_counts=True)
        for i, t in enumerate(unique_targets):
            if target_counts[i] < args.n_splits:
                cur_features = features[targets == t]
                new_features, j = [], 0
                while len(new_features) < args.n_splits - target_counts[i]:
                    if len(new_features) == 0:
                        new_features = np.expand_dims(cur_features[j], axis=0)
                    else:
                        new_features = np.concatenate((new_features, np.expand_dims(cur_features[j], axis=0)))
                    j += 1

                    if j == len(cur_features):
                        j = 0

                features = np.concatenate((features, new_features))
                targets = np.concatenate((targets, np.full(len(new_features), t)))

        if scaler_type == 'standard':
            scaler = StandardScaler().fit(features)
            features = scaler.transform(features)
            return features, targets, scaler
        else:
            return features, targets, None
    else:
        if scaler != None:
            features = scaler.transform(features)
        return features, targets
            

def prepare_sample(args, sample):
    if sample == "DUMMY":
        raise Exception(
            "Trying to use an uninitialized 'dummy' batch. This usually indicates "
            "that the total number of batches is smaller than the number of "
            "participating GPUs. Try reducing the batch size or using fewer GPUs."
        )

    if sample is None or len(sample) == 0:
        return None

    if torch.cuda.is_available() and not args.cpu:
        sample = utils.move_to_cuda(sample)

    def apply_half(t):
        if t.dtype is torch.float32:
            return t.half()
        return t

    if args.fp16:
        sample = utils.apply_to_sample(apply_half, sample)

    return sample

def valid_step(args, sample, task, model, criterion, dummy_batch, logger, raise_oom=False):
    """Do forward pass in evaluation mode."""

    with torch.no_grad():
        model.eval()
        criterion.eval()

        sample = prepare_sample(args, sample)
        if sample is None:
            sample = prepare_sample(args, dummy_batch)
            is_dummy_batch = True
        else:
            is_dummy_batch = False

        try:
            _loss, sample_size, logging_output = task.valid_step(
                sample, model, criterion
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                log_oom(e, logger)
                if not raise_oom:
                    logger.warning(
                        "ran out of memory in validation step, retrying batch"
                    )
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad = None  # free some memory
                    if torch.cuda.is_available() and not args.cpu:
                        torch.cuda.empty_cache()
                    return valid_step(args, sample, task, model, criterion, dummy_batch, logger, raise_oom=True)
            raise e

        logging_outputs = [logging_output]
        if is_dummy_batch:
            sample_size *= 0  # multiply by 0 to preserve device

    # gather logging outputs from all replicas
    if args.distributed_world_size > 1:
        logging_outputs, (sample_size, ) = aggregate_logging_outputs(
            args, task, criterion, logging_outputs, sample_size, ignore=is_dummy_batch,
        )

    # log validation stats
    logging_output = reduce_and_log_stats(args, task, criterion, logging_outputs, sample_size)

    return logging_output

def log_oom(exc, logger):
    msg = "OOM: Ran out of memory with exception: {}".format(exc)
    logger.warning(msg)
    if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
        for device_idx in range(torch.cuda.device_count()):
            logger.warning(torch.cuda.memory_summary(device=device_idx))
    sys.stderr.flush()

def aggregate_logging_outputs(
    args,
    task,
    criterion,
    logging_outputs: List[Dict[str, Any]],
    *extra_stats_to_sum,
    ignore=False,
):
    if task.__class__.logging_outputs_can_be_summed(criterion):
        return fast_stat_sync_sum(
            args, logging_outputs, *extra_stats_to_sum, ignore=ignore
        )
    else:
        return all_gather_list_sync(
            args, logging_outputs, *extra_stats_to_sum, ignore=ignore
        )

def fast_stat_sync_sum(
    args,
    logging_outputs: List[Dict[str, Any]],
    *extra_stats_to_sum,
    ignore=False,
):
    """
    Sync logging outputs across workers. fast_stat_sync_sum is
    faster than all_gather_list_sync, but is only suitable when
    logging outputs are scalars and can be summed. Note that
    *logging_outputs* cannot contain any nested dicts/lists.
    """
    data = {}
    for i, stat in enumerate(extra_stats_to_sum):
        data['extra_stats_' + str(i)] = stat
    if len(logging_outputs) > 0:
        log_keys = list(logging_outputs[0].keys())
        for k in log_keys:
            if not ignore:
                v = sum(log[k] for log in logging_outputs if k in log)
            else:
                v = logging_outputs[0][k]
                v = torch.zeros_like(v) if torch.is_tensor(v) else 0
            data['logging_outputs_' + k] = v
    else:
        log_keys = None

    data = distributed_utils.all_reduce_dict(
        data,
        device=args.device_id,
    )

    extra_stats_to_sum = [
        data['extra_stats_' + str(i)] for i in range(len(extra_stats_to_sum))
    ]
    if log_keys is not None:
        logging_outputs = [{k: data['logging_outputs_' + k] for k in log_keys}]
    else:
        logging_outputs = []
    return logging_outputs, extra_stats_to_sum

def all_gather_list_sync(
    args,
    logging_outputs: List[Dict[str, Any]],
    *extra_stats_to_sum,
    ignore=False,
):
    """
    Sync logging outputs across workers. all_gather_list_sync is
    suitable when logging outputs are complex types.
    """
    if ignore:
        logging_outputs = []
    results = list(zip(
        *distributed_utils.all_gather_list(
            [logging_outputs] + list(extra_stats_to_sum),
            max_size=getattr(args, 'all_gather_list_size', 16384),
        )
    ))
    logging_outputs, extra_stats_to_sum = results[0], results[1:]
    logging_outputs = list(chain.from_iterable(logging_outputs))
    extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
    return logging_outputs, extra_stats_to_sum

def reduce_and_log_stats(args, task, criterion, logging_outputs, sample_size, grad_norm=None):
    if grad_norm is not None:
        metrics.log_speed("ups", 1., priority=100, round=2)
        metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
        if args.clip_norm > 0:
            metrics.log_scalar(
                "clip",
                torch.where(
                    grad_norm > args.clip_norm,
                    grad_norm.new_tensor(100),
                    grad_norm.new_tensor(0),
                ),
                priority=500,
                round=1,
            )

    with metrics.aggregate() as agg:
        if logging_outputs is not None:
            task.reduce_metrics(logging_outputs, criterion)

        # support legacy interface
        logging_output = agg.get_smoothed_values()
        logging_output["sample_size"] = sample_size
        for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
            if key_to_delete in logging_output:
                del logging_output[key_to_delete]
        return logging_output

def get_ft_train_stats(stats):
    new_stats = copy.deepcopy(stats)
    eval_keys = ['acc', 'f1']
    for stat in stats:
        if sum([e in stat for e in eval_keys]) == 0:
            del new_stats[stat]
    return new_stats

def get_ft_valid_stats(args, stats, num_updates):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = num_updates
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats

def get_sklearn_stats(stats, num_updates):
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    stats['num_updates'] = num_updates
    return stats