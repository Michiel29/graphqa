import argparse
import json

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', type=str, help='path to JSON file of experiment configurations')
    return parser

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)    
