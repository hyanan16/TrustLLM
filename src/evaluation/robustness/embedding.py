import openai
import os
from utils import *


def get_embedding(text: str, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def run_get_embedding(root_dir, save_dir):
    folders_list = get_models()
    for folder in folders_list:
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(folder_path, file_name)
                    out_file_path = os.path.join(os.path.join(save_dir, folder), 'after_' + file_name)
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        for item in data:
                            item['embedding'] = get_embedding(item['res'])
                        with open(out_file_path, 'w') as json_file:
                            json.dump(data, json_file, ensure_ascii=False, indent=4)
                        print('Done!' + out_file_path)


if __name__ == '__main__':
    pass
