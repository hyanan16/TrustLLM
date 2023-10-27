import json


def get_models(filename='../../config/evaluation_config.json'):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config['robustness']['naturalnoise']['models']