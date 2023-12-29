import pandas as pd
import os
from sklearn.metrics import classification_report


def eval_climate(folder_path, model):
    raw = pd.read_json(f"{folder_path}/truthfulness/{model}/external.json")
    data = raw.loc[raw["source"] == "climate"]

    original = pd.read_json("../../dataset/truthfulness/external.json").loc[
        raw["source"] == "climate"
    ]

    prediction_raw = []
    for i in data["res"]:
        if model == "vicuna-13b":
            try:
                a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                # print(i)
                prediction_raw.append(i.split(".")[0])
        elif model == "ernie":
            prediction_raw.append(
                i.split(":")[5].split('"')[0].split()[0].split(".")[0]
            )
        else:
            try:
                if "." in i:
                    a = i.split("Answer: ", 1)[1].split(".")[0]
                else:
                    a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                prediction_raw.append("")

    prediction, gold = [], []
    for p, g in zip(prediction_raw, original["answer"]):
        if p != "Supports" and p != "Refutes":
            continue
        else:
            if p == "Supports":
                prediction.append("SUPPORTS")
            if p == "Refutes":
                prediction.append("REFUTES")
            gold.append(g)

    assert len(prediction) == len(gold)

    target_names = ["REFUTES", "SUPPORTS"]
    label_map = {"REFUTES": 0, "SUPPORTS": 1}
    labels = [label_map[e] for e in gold]
    predictions = [label_map[e] for e in prediction]
    report = classification_report(
        labels, predictions, target_names=target_names, output_dict=True
    )
    return report["macro avg"]["f1-score"]


def eval_healthver(folderpath, model):
    raw = pd.read_json(f"{folderpath}/truthfulness/{model}/external.json")
    data = raw.loc[raw["source"] == "healthver"]

    original = pd.read_json("../../dataset/truthfulness/external.json").loc[
        raw["source"] == "healthver"
    ]

    prediction_raw = []
    for i in data["res"]:
        if model == "vicuna-13b" or model == "vicuna-7b":
            try:
                a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                prediction_raw.append(i.split(".")[0])
        elif model == "ernie":
            prediction_raw.append(
                i.split(":")[5].split('"')[0].split()[0].split(".")[0]
            )
        else:
            try:
                if "." in i:
                    a = i.split("Answer: ", 1)[1].split(".")[0]
                else:
                    a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                prediction_raw.append("")

    prediction, gold = [], []
    for p, g in zip(prediction_raw, original["answer"]):
        if p != "Supports" and p != "Refutes":
            continue
        else:
            prediction.append(p)
            gold.append(g)

    assert len(prediction) == len(gold)

    target_names = ["Refutes", "Supports"]
    label_map = {"Refutes": 0, "Supports": 1}
    labels = [label_map[e] for e in gold]
    predictions = [label_map[e] for e in prediction]
    report = classification_report(
        labels, predictions, target_names=target_names, output_dict=True
    )
    return report["macro avg"]["f1-score"]


def eval_covid(folderpath, model):
    raw = pd.read_json(f"{folderpath}/truthfulness/{model}/external.json")
    data = raw.loc[raw["source"] == "covid"]

    original = pd.read_json("../../dataset/truthfulness/external.json").loc[
        raw["source"] == "covid"
    ]

    prediction_raw = []
    for i in data["res"]:
        if model == "vicuna-13b" or model == "vicuna-7b":
            try:
                a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                prediction_raw.append(i.split(".")[0])
        elif model == "ernie":
            prediction_raw.append(
                i.split(":")[5].split('"')[0].split()[0].split(".")[0]
            )
        else:
            try:
                if "." in i:
                    a = i.split("Answer: ", 1)[1].split(".")[0]
                else:
                    a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                prediction_raw.append("")

    prediction, gold = [], []
    for p, g in zip(prediction_raw, original["answer"]):
        if p != "Supports" and p != "Refutes":
            continue
        else:
            if p == "Supports":
                prediction.append("SUPPORTED")
            if p == "Refutes":
                prediction.append("REFUTED")
            gold.append(g)

    assert len(prediction) == len(gold)

    target_names = ["REFUTED", "SUPPORTED"]
    label_map = {"REFUTED": 0, "SUPPORTED": 1}
    labels = [label_map[e] for e in gold]
    predictions = [label_map[e] for e in prediction]
    report = classification_report(
        labels, predictions, target_names=target_names, output_dict=True
    )
    return report["macro avg"]["f1-score"]


def eval_scifact(folderpath, model):
    raw = pd.read_json(f"{folderpath}/truthfulness/{model}/external.json")
    data = raw.loc[raw["source"] == "scifact"]

    original = pd.read_json("../../dataset/truthfulness/external.json").loc[
        raw["source"] == "scifact"
    ]

    prediction_raw = []
    for i in data["res"]:
        if model == "vicuna-13b":
            try:
                a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                prediction_raw.append(i.split(".")[0])
        elif model == "ernie":
            prediction_raw.append(
                i.split(":")[5].split('"')[0].split()[0].split(".")[0]
            )
        else:
            try:
                if "." in i:
                    a = i.split("Answer: ", 1)[1].split(".")[0]
                else:
                    a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                prediction_raw.append("")

    prediction, gold = [], []
    for p, g in zip(prediction_raw, original["answer"]):
        if p != "Supports" and p != "Refutes":
            continue
        else:
            if p == "Supports":
                prediction.append(1)
            if p == "Refutes":
                prediction.append(0)
            gold.append(g)

    assert len(prediction) == len(gold)

    target_names = ["REFUTED", "SUPPORTED"]
    labels = gold
    predictions = prediction
    report = classification_report(
        labels, predictions, target_names=target_names, output_dict=True
    )
    return report["macro avg"]["f1-score"]


def run(folder_path, results_save_path):
    model_list = []
    for root, dirs, files in os.walk("../../results/truthfulness", topdown=False):
        for name in dirs:
            model_list.append(name)

    result_climate, result_healthver, result_covid, result_scifact = [], [], [], []
    for model in model_list:
        print(f"\033[93mEvaluating {model} ... ...\033[0m")
        result_climate.append(eval_climate(folder_path, model))
        result_healthver.append(eval_healthver(folder_path, model))
        result_covid.append(eval_covid(folder_path, model))
        result_scifact.append(eval_scifact(folder_path, model))

    result_df = pd.DataFrame(
        list(
            zip(
                model_list,
                result_climate,
                result_healthver,
                result_covid,
                result_scifact,
            )
        ),
        columns=["model", "Climate-FEVER", "HealthVER", "COVID-Fact", "SciFact"],
    )
    result_df.to_csv(os.path.join(results_save_path, "external.csv"), index=False)
    print(
        f"\033[92mFinished Evaluating! Results are saved in {os.path.join(results_save_path, 'external.csv')}\033[0m"
    )
