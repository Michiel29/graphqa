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
            setattr(namespace, key, value)

        


