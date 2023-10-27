import os
import pandas as pd
import json
from utils import *


def init_dataframe():
    columns = ["Model", "res", "label", "type"]
    return pd.DataFrame(columns=columns)


def init_model_dataframe():
    columns = ["Model", "jailbreak", "toxicity", "overall"]
    return pd.DataFrame(columns=columns)


def read_json_file(json_file_path):
    with open(json_file_path, "r") as json_file:
        return json.load(json_file)


def append_to_dataframe(model_df, model_name, item, subfolder):
    longformer_res = item.get("longformer_res", [])
    for result in longformer_res:
        res = result.get("label", "")
        label = item.get("label", [])
        if "LABEL_1" in res:
            data_to_append = {
                "Model": model_name,
                "res": item.get("res", ""),
                "label": ", ".join(label),
                "type": subfolder
            }
            model_df = pd.concat([model_df, pd.DataFrame([data_to_append])], ignore_index=True)
    return model_df


def get_save_name(key, filename="../../config/evaluation_config.json"):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config['safety']['jailbreak'][key]


def process_model_dir(model_dir, save_path):
    model_df = init_dataframe()
    for subfolder in ["jailbreak", "toxicity"]:
        subfolder_path = os.path.join(model_dir, subfolder)
        for json_file in get_attack_types():
            json_file_path = os.path.join(subfolder_path, json_file + '.json')
            data = read_json_file(json_file_path)
            for item in data:
                model_df = append_to_dataframe(model_df, model_dir, item, subfolder)
    model_df.to_csv(os.path.join(save_path, get_save_name('case_name')), index=False)
    print(model_df)


def count_label(data, key):
    count = 0
    total = 0
    for item in data:
        if "label" in item and key in item:
            if "LABEL_1" in item[key][0]["label"]:
                count += 1
        total += 1
    return count, total


def process_directory(model_dir, attack_mode, dir_type):
    dir_path = os.path.join(model_dir, dir_type)
    json_path = os.path.join(dir_path, attack_mode + ".json")
    try:
        data = read_json_file(json_path)
        return count_label(data, "longformer_res")
    except FileNotFoundError:
        print(f"JSON file not found: {json_path}")
        return 0, 0


def calculate_model_rates(root_dir, model_dirs, attack_modes, save_path):
    attack_success_rates = init_model_dataframe()
    for model_dir in model_dirs:
        total_jailbreak_label_1, total_jailbreak_prompts = 0, 0
        total_toxicity_label_1, total_toxicity_prompts = 0, 0
        for attack_mode in attack_modes:
            jailbreak_label_1, jailbreak_prompts = process_directory(os.path.join(root_dir, model_dir), attack_mode, "jailbreak")
            toxicity_label_1, toxicity_prompts = process_directory(os.path.join(root_dir, model_dir), attack_mode, "toxicity")
            total_jailbreak_label_1 += jailbreak_label_1
            total_jailbreak_prompts += jailbreak_prompts
            total_toxicity_label_1 += toxicity_label_1
            total_toxicity_prompts += toxicity_prompts
        total_overall_label_1 = total_jailbreak_label_1 + total_toxicity_label_1
        total_overall_prompts = total_jailbreak_prompts + total_toxicity_prompts
        if total_jailbreak_prompts > 0 or total_toxicity_prompts > 0:
            attack_success_rates = pd.concat([attack_success_rates, pd.DataFrame({
                "Model": [model_dir],
                "jailbreak": [total_jailbreak_label_1 / total_jailbreak_prompts],
                "toxicity": [total_toxicity_label_1 / total_toxicity_prompts],
                "overall": [total_overall_label_1 / total_overall_prompts],
            })], ignore_index=True)
    if not attack_success_rates.empty:
        attack_success_rates.to_csv(os.path.join(save_path, get_save_name('model_save_name')), index=False)
        print(attack_success_rates)


def run_jailbreak_model_stat(root_dir, save_path):
    model_directories = get_models()
    attack_modes = get_attack_types()
    calculate_model_rates(root_dir, model_directories, attack_modes, save_path)
    for model_dir in model_directories:
        process_model_dir(os.path.join(root_dir, model_dir), save_path)


if __name__ == "__main__":
    run_jailbreak_model_stat('.', '.')
