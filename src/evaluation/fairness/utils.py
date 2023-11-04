import json

import os,json

def read_json(file):
    with open(file,"r") as f:
        data=json.load(f)
        return data

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
        
def get_models(key, filename='../../config/evaluation_config.json'):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config['fairness'][key]['models']


