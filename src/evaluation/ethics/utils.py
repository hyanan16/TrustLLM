import json


def get_models(type, filename='../../config/evaluation_config.json'):
    with open(filename) as f:
        data = json.load(f)
    return data['ethics'][type]['models']