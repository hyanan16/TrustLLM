import os
import pandas as pd
import json


def get_models(filename='../../config/evaluation_config.json'):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config['safety']['jailbreak']['models']


def init_dataframe():
    columns = ["Model", "res", "label", "type"]
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


def get_attack_types(filename='../../config/evaluation_config.json'):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config['safety']['jailbreak']['types']


def get_save_name(filename="../../config/evaluation_config.json"):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config['safety']['jailbreak']['model_save_name']


def process_model_dir(model_dir, save_path):
    model_df = init_dataframe()
    for subfolder in ["jailbreak", "toxicity"]:
        subfolder_path = os.path.join(model_dir, subfolder)
        for json_file in get_attack_types():
            json_file_path = os.path.join(subfolder_path, json_file + '.json')
            data = read_json_file(json_file_path)
            for item in data:
                model_df = append_to_dataframe(model_df, model_dir, item, subfolder)
    model_df.to_csv(os.path.join(save_path, get_save_name()), index=False)
    print(model_df)


def run_jailbreak_model_stat(root_dir, save_path):
    model_directories = get_models()
    for model_dir in model_directories:
        process_model_dir(os.path.join(root_dir, model_dir), save_path)


if __name__ == "__main__":
    run_jailbreak_model_stat('.','output.csv')
