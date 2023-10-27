import json


def get_models(filename='../../config/evaluation_config.json'):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config['safety']['jailbreak']['models']


def get_attack_types(filename='../../config/evaluation_config.json'):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config['safety']['jailbreak']['types']