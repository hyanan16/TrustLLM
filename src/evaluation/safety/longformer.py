import os
import json
from tqdm import tqdm
from transformers import pipeline
from utils import *

# Usage example:
# evaluate_with_longformer('./TrustLLM/Safety/Misuse/test_res', './TrustLLM/Safety/Misuse/eval_res', 'LibrAI/longformer-harmful-ro')

def get_longformer_model_path(filename='../../config/evaluation_config.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['safety']['longformer_path']


def evaluate_with_longformer(input_path, save_path, model_path=get_longformer_model_path()):
    """
    Evaluate data with the Longformer model.

    Parameters:
    - input_path (str): Path to the directory containing the JSON files to be processed.
    - save_path (str): Path to the directory where the processed files will be saved.
    - model_path (str): Huggingface model path.

    Returns:
    None. Processed JSON files are saved to the save_path directory.
    """
    dirs = ['jailbreak', 'toxicity']
    models = get_models()
    types = get_attack_types()
    classifier = pipeline(model=model_path,device='cuda:0')
    for dir in dirs:
        for model in models:
            for type in types:
                all_data = []
                with open(os.path.join(input_path, dir, model, type + '.json'), 'r') as f:
                    data = json.load(f)
                    for el in tqdm(data):
                        text = el['res']
                        res = classifier(text)
                        el['longformer_res'] = res
                        all_data.append(el)
                save_dir = os.path.join(save_path, dir, model)
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, type + '.json'), 'w') as f2:
                    json.dump(all_data, f2)


