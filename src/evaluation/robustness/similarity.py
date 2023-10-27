import os
import json
from sklearn.metrics.pairwise import cosine_similarity

# base_dir: initial base folder
# output_dir: output folder for json file with similarity and embedding
# output_dir_similarity: get folder containing only json file with similarity
base_dir = './AdvInstruction/new'
output_dir = './AdvInstruction/new_similarity'
output_dir_similarity = './AdvInstruction/similarity'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def similarity(embedding_1, embedding_2):
    embedding_1 = [embedding_1]
    embedding_2 = [embedding_2]
    similarity = cosine_similarity(embedding_1, embedding_2)
    return similarity[0][0] 


def similarity_folder(output_dir=output_dir, output_dir_similarity=output_dir_similarity):
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


for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
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
                            original_item = next((x for x in data if x['type'] == 'original' and x['index'] == item['index']), None)
                            if original_item:
                                item['similarity'] = similarity(item['embedding'], original_item['embedding'])
                output_filename = folder + "_" + filename.replace('after_', '')
                output_file_path = os.path.join(output_dir, output_filename)
                with open(output_file_path, 'w') as f:
                    json.dump(data, f, indent=4)


similarity_folder()