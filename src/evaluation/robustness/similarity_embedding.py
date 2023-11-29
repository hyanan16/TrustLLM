import os
import json

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import openai
from utils import get_models


# Function to get the embedding of a text using the specified model
def get_embedding(text: str, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


# Function to compute similarity between two embeddings
def similarity(embedding_1, embedding_2):
    similarity = cosine_similarity([embedding_1], [embedding_2])
    return similarity[0][0]


# Function to create a directory for similarity output files
def create_similarity_folder(output_dir, output_dir_similarity):
    if not os.path.exists(output_dir_similarity):
        os.makedirs(output_dir_similarity)
    for file in os.listdir(output_dir):
        json_path = os.path.join(output_dir, file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            for item in data:
                del item['embedding']
        output_file_path = os.path.join(output_dir_similarity, file)
        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=4)


# Function to process JSON files, calculate embeddings, and store the results
def process_json_files(root_dir, save_dir):
    folders_list = get_models('naturalnoise')
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


# Function to calculate similarities and save results
def calculate_and_save_similarities(root_dir, embedding_path, save_dir):
    for folder in get_models('naturalnoise'):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.startswith('after_'):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        for item in data:
                            if item['type'] == 'original':
                                item['similarity'] = 1.0
                            else:
                                original_item = next(
                                    (x for x in data if x['type'] == 'original' and x['index'] == item['index']), None)
                                if original_item:
                                    item['similarity'] = similarity(item['embedding'], original_item['embedding'])
                    output_filename = folder + "_" + filename.replace('after_', '')
                    output_file_path = os.path.join(embedding_path, output_filename)
                    with open(output_file_path, 'w') as f:
                        json.dump(data, f, indent=4)
    create_similarity_folder(embedding_path, save_dir)


def gen_advinstruction_res(dir_path, save_path):
    file_list = os.listdir(dir_path)
    model_res = {}

    for file in file_list:
        model = file.split('_')[0]
        with open(os.path.join(dir_path, file), 'r') as f:
            data = json.load(f)
            for el in data:
                attack_type = el['type']
                if attack_type == "original":
                    continue
                if attack_type not in model_res:
                    model_res[attack_type] = {}
                if model not in model_res[attack_type]:
                    model_res[attack_type][model] = []
                model_res[attack_type][model].append(el['similarity'])

    # Calculate average for each model and attack type
    for attack_type, models in model_res.items():
        for model, similarities in models.items():
            model_res[attack_type][model] = sum(similarities) / len(similarities)

    # Create DataFrame and calculate row averages
    df = pd.DataFrame(model_res)
    df['Average'] = df.mean(axis=1)

    # Save DataFrame to XLSX
    df.to_excel(os.path.join(save_path, 'model_res_advinstruction.xlsx'), index_label='Attack Type')


if __name__ == '__main__':
    pass
