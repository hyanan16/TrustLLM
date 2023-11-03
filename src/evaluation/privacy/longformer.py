import json
import os
from tqdm import tqdm
from transformers import pipeline


def get_longformer_model_path(filename='../../config/evaluation_config.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['privacy']['longformer_path']

classifier = pipeline(model=get_longformer_model_path())


def run_privacy_longformer(fire_path, save_path):
    for root, dirs, files in os.walk(fire_path):
        for file in files:
            if file.endswith('.json'):
                print(os.path.join(root, file))
                all_data = []
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    for el in tqdm(data):
                        text = el['res']
                        res = classifier(text)
                        el['longformer_res'] = res
                        all_data.append(el)
                save_dir = os.path.join(save_path, os.path.relpath(root, fire_path))
                os.makedirs(save_dir, exist_ok=True)

                with open(os.path.join(save_dir, file), 'w') as f2:
                    json.dump(all_data, f2)


if __name__ == "__main__":
    run_privacy_longformer(".", ".")
