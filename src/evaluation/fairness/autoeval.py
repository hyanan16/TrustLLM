
import os


import os
import pandas as pd
import traceback

from utils import *


models = get_models()

base_dir = '' #faieness/test_res/
results = []



for model in models:
# for model in ['llama2-70b']:
    try:
        files = os.listdir(base_dir + model)
        for file in files:
            if os.path.basename(file)=='implicit_ETHICS.json':
                file_path = os.path.join(base_dir, model, file)
                all_res = evaluate_matches(file_path)
                save_path = os.path.join(base_dir, model, 'implicit_ETHICS_gptres.json')
                save_json(all_res,save_path)
    except Exception as e:
        print(traceback.format_exc())

