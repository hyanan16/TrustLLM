import os
import json
from tqdm import tqdm
from transformers import pipeline

def evaluate_with_longformer(input_path, save_path, model_path):
    """
    Evaluate data with the Longformer model.

    Parameters:
    - input_path (str): Path to the directory containing the JSON files to be processed.
    - save_path (str): Path to the directory where the processed files will be saved.
    - model_path (str): Huggingface model path.

    Returns:
    None. Processed JSON files are saved to the save_path directory.
    """
    
    classifier = pipeline(model=model_path)

    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.json'):
                print(os.path.join(root, file)) 
                all_data = []
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    for el in tqdm(data):
                        if el['label'] != "['arithmetical']":
                            text = el['res']
                            res = classifier(text)
                            el['longformer_res'] = res
                            all_data.append(el)
                save_dir = os.path.join(save_path, os.path.relpath(root, input_path))
                os.makedirs(save_dir, exist_ok=True)

                with open(os.path.join(save_dir, file), 'w') as f2:
                    json.dump(all_data, f2)

# Usage example:
# evaluate_with_longformer('./TrustLLM/Safety/Misuse/test_res', './TrustLLM/Safety/Misuse/eval_res', 'LibrAI/longformer-harmful-ro')
