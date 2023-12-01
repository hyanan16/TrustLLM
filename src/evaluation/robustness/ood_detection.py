import os
import json
from utils import *
import pandas as pd


def ood_detection_res(root_dir, save_dir):
    models = get_models('ood')
    files = os.listdir(root_dir)
    data_stat = {}
    for file in files:
        if file in models:
            with open(os.path.join(root_dir, file, 'ood_detection.json'), 'r') as f:
                model_data = json.load(f)
                total_length = len(model_data)
                if file not in data_stat:
                    data_stat[file] = 0
                for el in model_data:
                    if el['longformer_res'][0]['label'] == 'LABEL_0':
                        data_stat[file] += 1
            data_stat[file] = data_stat[file] / total_length
    df = pd.DataFrame.from_dict(data_stat, orient='index')
    df.to_csv(os.path.join(save_dir, 'ood_detection_res.csv'))


if __name__ == '__main__':
    ood_detection_res('../../../results/robustness/ood', '../../../results')


