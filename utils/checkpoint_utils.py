from datetime import datetime
import os

import numpy as np

def select_component_state(model_state_dict, prefix):
    """Returns new state dict with only model parameters starting with prefix"""
    component_state_dict = {key: value for key, value in model_state_dict.items() if key.startswith(prefix)}
    return component_state_dict

def generate_save_dir(args, system_args):
    """For new experiments, generate checkpointing directory of form task/architecture/lr/datetime. When restoring from a checkpoint, return the path with the latest datetime in task/architecture/lr."""

    restore_file = getattr(args, 'restore_file', False)

    new_save_base = args.save_dir

    save_attribute_names = args.path_attributes + [
        arg_name.strip('-').replace('-', '_') for arg_name in system_args
    if
        arg_name.startswith('-')
        and arg_name not in args.path_attributes
        and not arg_name == '--config'
    ]


    for attribute_name in save_attribute_names:
        attribute_value = getattr(args, attribute_name)
        if isinstance(attribute_value, list):
            attribute_string = '__'.join([str(val) for val in attribute_value])
        else:
            attribute_string = str(attribute_value)

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