import openai
import os
import json
from tqdm import tqdm
import utils

openai.api_key = ''

def get_embedding(text: str, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def embedding_dir(base_dir):
    model_list = utils.get_models('naturalnoise')
    for model in model_list:
        folder_path = os.path.join(base_dir, model)
        print(folder_path)
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, 'AdvInstruction.json')
            out_file_path = os.path.join(folder_path, 'after_' + 'AdvInstruction.json')
            with open(file_path, 'r') as json_file:
                print('Begin to process ' + file_path)
                data = json.load(json_file)
                for item in tqdm(data):
                    item['embedding'] = get_embedding(item['res'])
                with open(out_file_path, 'w') as json_file:
                    json.dump(data, json_file, ensure_ascii=False, indent=4)
                print('Done!' + out_file_path)