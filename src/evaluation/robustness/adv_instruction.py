import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_models


def similarity(embedding_1, embedding_2):
    embedding_1 = [embedding_1]
    embedding_2 = [embedding_2]
    similarity = cosine_similarity(embedding_1, embedding_2)
    return similarity[0][0]


def del_embedding(output_dir):
    for file in os.listdir(output_dir):
        json_path = os.path.join(output_dir, file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            for item in data:
                del item['embedding']
        output_file_path = os.path.join(output_dir, file)
        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=4)


def get_similarity(base_dir, output_dir):
    models = get_models('natrualnoise')
    for model in models:
        folder_path = os.path.join(base_dir, model)
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
                    output_filename = model + "_" + filename.replace('after_', '')
                    output_file_path = os.path.join(output_dir, output_filename)
                    with open(output_file_path, 'w') as f:
                        json.dump(data, f, indent=4)
    del_embedding(output_dir)


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
    for attack_type, models in model_res.items():
        for model, similarities in models.items():
            model_res[attack_type][model] = sum(similarities) / len(similarities)
    df = pd.DataFrame(model_res)
    df['Average'] = df.mean(axis=1)
    df.to_excel(os.path.join(save_path, 'model_res_advinstruction.xlsx'), index_label='Attack Type')


def adv_instruction_eval(root_dir, save_path):
    if not os.path.exists(os.path.join(root_dir, 'robustness_similarity')):
        os.makedirs(os.path.join(root_dir, 'robustness_similarity'))
    get_similarity(root_dir, os.path.join(root_dir, 'robustness_similarity'))
    gen_advinstruction_res(os.path.join(root_dir, 'robustness_similarity'), save_path)


if __name__ == '__main__':
    gen_advinstruction_res('../../../results/robustness/robustness_similarity', '../../../results')
