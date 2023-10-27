import os
import json
import pandas as pd


def read_json_data(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            return json.load(json_file)
    return []


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
    data = read_json_data(json_path)
    return count_label(data, "longformer_res")


def calculate_rates(model_dirs, attack_modes):
    attack_success_rates = {}

    for attack_mode in attack_modes:
        total_jailbreak_label_1, total_jailbreak_prompts = 0, 0
        total_toxicity_label_1, total_toxicity_prompts = 0, 0

        for model_dir in model_dirs:
            jailbreak_label_1, jailbreak_prompts = process_directory(model_dir, attack_mode, "safety")
            toxicity_label_1, toxicity_prompts = process_directory(model_dir, attack_mode, "toxicity")

            total_jailbreak_label_1 += jailbreak_label_1
            total_jailbreak_prompts += jailbreak_prompts
            total_toxicity_label_1 += toxicity_label_1
            total_toxicity_prompts += toxicity_prompts

        total_overall_label_1 = total_jailbreak_label_1 + total_toxicity_label_1
        total_overall_prompts = total_jailbreak_prompts + total_toxicity_prompts

        if total_jailbreak_prompts > 0 or total_toxicity_prompts > 0:
            attack_success_rates[attack_mode] = {
                "label": attack_mode,
                "jailbreak": total_jailbreak_label_1 / total_jailbreak_prompts,
                "toxicity": total_toxicity_label_1 / total_toxicity_prompts,
                "overall": total_overall_label_1 / total_overall_prompts,
            }

    return attack_success_rates


def save_to_csv(data, filename):
    df = pd.DataFrame(data).transpose()
    print(df.to_string(index=False))
    df.to_csv(filename, index=False)


def get_attack_types(filename='../../config/evaluation_config.json'):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config['safety']['jailbreak']['types']


def get_models(filename='../../config/evaluation_config.json'):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config['safety']['jailbreak']['models']


def run_jailbreak_stat(savename):
    model_dirs = get_models()
    attack_modes = get_attack_types()
    attack_success_rates = calculate_rates(model_dirs, attack_modes)
    save_to_csv(attack_success_rates, savename)


if __name__ == "__main__":
    run_jailbreak_stat("label_success_rates.csv")
