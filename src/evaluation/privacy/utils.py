import json


def get_models(key, filename='../../config/evaluation_config.json'):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config['privacy'][key]['models']

