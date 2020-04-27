from datetime import datetime
import os
import regex as re

import numpy as np

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


def get_training_name(args):
    s = [get_task_str(args), get_model_str(args)]
    if len(args.exp_name) > 0:
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
        new_save_base = os.path.join(new_save_base, attribute_name + '_' + attribute_string)

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