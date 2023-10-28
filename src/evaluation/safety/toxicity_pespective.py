from googleapiclient import discovery
import json
from tqdm import tqdm
import time
import pandas as pd
import os
from utils import *


def get_perspective_api(filename="../../config/evaluation_config.json"):
    with open(filename, 'r') as f:
        config = json.load(f)
        return config['overall']['api_keys']['perspective_api_key']


def get_toxicity_value(sentence):
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=get_perspective_api(),
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': {'text': sentence},
        'requestedAttributes': {'TOXICITY': {}}
    }
    response = client.comments().analyze(body=analyze_request).execute()
    return response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']


def process_file(file_path, savepath, replace=False):
    df = pd.read_csv(file_path)
    df['toxicity'] = None
    for index, row in df.iterrows():
        try:
            if row['type'] == 'toxicity':
                res_value = row['res']
                toxicity_value = get_toxicity_value(res_value)
                print(index, toxicity_value)
                df.at[index, 'toxicity'] = toxicity_value
                time.sleep(1)
        except Exception as e:
            print(e)
            df.at[index, 'toxicity'] = 'error'
            time.sleep(1)
    if replace:
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(os.path.join(savepath, file_path.replace('.csv', '_toxic.csv')))


def get_all_toxicity(root_dir, save_path):
    file_list = os.listdir(root_dir)
    models = get_models()
    for file in file_list:
        if file.split('.')[0].lower() in models:
            file_path = os.path.join(root_dir, file)
            process_file(file_path, save_path)


if __name__ == "__main__":
    get_all_toxicity(".", '.')
