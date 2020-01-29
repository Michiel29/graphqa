import argparse
import json

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)    

def update_namespace(namespace, input_dict):
    for key, value in input_dict.items():
        if isinstance(value, dict):
            update_namespace(namespace, value)
        else:
            if not hasattr(namespace, key):
                setattr(namespace, key, value)

def modify_factory(config):

    def parser_modifier(parser):
        for action in parser._actions:
            if action.dest in config:
                setattr(action, 'default', config[action.dest])
    
    return parser_modifier            

        


