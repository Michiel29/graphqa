import os
import argparse
import commentjson as json

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def update_namespace(namespace, input_dict):
    """Add dictionary to argparse namespace. Only adds arguments that are not already in parser"""
    for key, value in input_dict.items():
        if not hasattr(namespace, key):
            setattr(namespace, key, value)

def modify_factory(config):
    """Overrides defaults of argparser with config values where applicable"""
    def parser_modifier(parser):
        for action in parser._actions:
            if action.dest in config:
                setattr(action, 'default', config[action.dest])

    return parser_modifier


def update_config(config1, config2, override=False):
    new_config = {}
    if override:
        order = [config1, config2]
    else:
        order = [config2, config1]
    for config in order:
        new_config.update(config)

    return new_config


def compose_configs(path):

    is_dir = os.path.isdir(path)
    if is_dir:
        path = os.path.join(path, 'default.json')

    config = read_json(path)

    add_config_paths = []
    if 'add_configs' in config:
        add_config_paths = config['add_configs']

    base_path = os.path.split(path)[0]
    for config_path in add_config_paths:
        is_relative = (config_path[0] != '/')
        if is_relative:
            config_path = os.path.join(base_path, config_path)

        add_config = compose_configs(config_path)
        config = update_config(config, add_config)

    return config



def save_config(config, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config_output_path = os.path.join(save_dir, 'config.json')
    with open(config_output_path, 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)
