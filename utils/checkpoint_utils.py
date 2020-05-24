import collections
import logging
from datetime import datetime
import os
import regex as re

import numpy as np

from fairseq.checkpoint_utils import checkpoint_paths, PathManager

logger = logging.getLogger(__name__)


def select_component_state(model_state_dict, prefix):
    """Returns new state dict with only model parameters starting with prefix"""
    component_state_dict = {key: value for key, value in model_state_dict.items() if key.startswith(prefix)}
    return component_state_dict


def get_task_str(args):
    if args.task != 'multi_task':
        return args.task
    else:
        return '_'.join(args.tasks.keys())


ARCH_SUBSTR_TO_SHORT_NAME = {
    'roberta_small': 'small',
    'roberta_base': 'base',
    'roberta_large': 'large',
}


def get_model_str(args):
    for k, v in ARCH_SUBSTR_TO_SHORT_NAME.items():
        if k in args.arch:
            return v
    else:
        return args.arch


def get_training_name(args, append_exp_name=True):
    s = [get_task_str(args), get_model_str(args)]
    if len(args.exp_name) > 0 and append_exp_name:
        s.append(args.exp_name)
    return '/'.join(s)


def get_attribute_value_str(args, attribute_name):
    if not hasattr(args, attribute_name):
        return None
    attribute_value = getattr(args, attribute_name)
    if isinstance(attribute_value, list):
        attribute_string = '__'.join([str(val) for val in attribute_value])
    else:
        attribute_string = str(attribute_value)
    return attribute_string


NEPTUNE_TAG_REGEX = '[^a-zA-Z0-9]'


def generate_tags(args):
    tags = []
    for attribute_name in args.tag_attributes:
        attribute_value = get_attribute_value_str(args, attribute_name)
        if attribute_value is not None:
            tag = re.sub(NEPTUNE_TAG_REGEX, '-', attribute_name + '-' + attribute_value)
            tags.append(tag)
    return tags


def generate_save_dir(args, training_name, system_args):
    """For new experiments, generate checkpointing directory of form task/architecture/lr/datetime. When restoring from a checkpoint, return the path with the latest datetime in task/architecture/lr."""

    restore_file = getattr(args, 'restore_file', False)

    assert '-' not in training_name

    new_save_base = os.path.join(args.save_dir, training_name)

    save_attribute_names = [
        arg_name.strip('-').replace('-', '_')
        for arg_name in system_args
        if arg_name.startswith('-') and arg_name not in ['--config', '--exp-name']
    ]

    for attribute_name in save_attribute_names:
        attribute_value = get_attribute_value_str(args, attribute_name)
        new_save_base = os.path.join(new_save_base, attribute_name + '_' + attribute_value)

    if restore_file:
        sub_dirs = next(os.walk(os.path.join(new_save_base,'.')))[1]

        assert len(sub_dirs) > 0
        time_stamps = [''.join(filter(str.isdigit, dirname)) for dirname in sub_dirs]
        latest_dir_idx = np.argsort(time_stamps)[-1]
        new_save_dir = os.path.join(new_save_base, sub_dirs[latest_dir_idx])
    else:
        dt_string = datetime.now().strftime("%mm_%dd_%Hh_%Mm_%Ss")
        new_save_dir = os.path.join(new_save_base, dt_string)

    return new_save_dir

def handle_state_dict_keys(missing_keys, unexpected_keys):
    if len(missing_keys) > 0:
        print('missing_keys: {}'.format(missing_keys))
        raise KeyError('missing state dict key')

    if len(unexpected_keys) > 0:
        print('unexpected_keys: {}'.format(unexpected_keys))



def save_checkpoint(args, trainer, epoch_itr, val_loss, save_extra_state):
    from fairseq import distributed_utils, meters

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if args.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if args.no_save or not distributed_utils.is_master(args):
        return

    def is_better(a, b):
        return a >= b if args.maximize_best_checkpoint_metric else a <= b

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}.pt".format(epoch)] = (
        end_of_epoch
        and not args.no_epoch_checkpoints
        and epoch % args.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}.pt".format(epoch, updates)] = (
        not end_of_epoch
        and args.save_interval_updates > 0
        and updates % args.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best.pt"] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(val_loss, save_checkpoint.best)
    )
    if val_loss is not None and args.keep_best_checkpoints > 0:
        checkpoint_conds["checkpoint.best_{}_{:.2f}.pt".format(
            args.best_checkpoint_metric, val_loss)] = (
            not hasattr(save_checkpoint, "best")
            or is_better(val_loss, save_checkpoint.best)
        )
    checkpoint_conds["checkpoint_last.pt"] = not args.no_last_checkpoints

    extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})
    extra_state.update(save_extra_state)

    checkpoints = [
        os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0:
        trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            PathManager.copy(checkpoints[0], cp, overwrite=True)

        write_timer.stop()
        logger.info(
            "saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            args.save_dir, pattern=r"checkpoint_\d+_(\d+)\.pt"
        )
        for old_chk in checkpoints[args.keep_interval_updates :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(args.save_dir, pattern=r"checkpoint(\d+)\.pt")
        for old_chk in checkpoints[args.keep_last_epochs :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_best_checkpoints > 0:
        # only keep the best N checkpoints according to validation metric
        checkpoints = checkpoint_paths(
            args.save_dir, pattern=r"checkpoint\.best_{}_(\d+\.?\d*)\.pt".format(args.best_checkpoint_metric))
        if not args.maximize_best_checkpoint_metric:
            checkpoints = checkpoints[::-1]
        for old_chk in checkpoints[args.keep_best_checkpoints:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)