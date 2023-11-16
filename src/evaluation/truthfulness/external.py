import pandas as pd
import argparse
import os
import re
import string
import pickle
from collections import Counter
from statistics import mean
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


def eval_climate(model):
    raw = pd.read_json(f"../../results/truthfulness/{model}/external.json")
    data = raw.loc[raw["source"] == "climate"]

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
                a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                prediction_raw.append("")

    data_gold = pd.read_json(f"../../dataset/truthfulness/Gold/climate_gold_2.0.json")

    prediction, gold = [], []
    for p, g in zip(prediction_raw, data_gold["answer"]):
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


def eval_healthver(model):
    raw = pd.read_json(f"../../results/truthfulness/{model}/external.json")
    data = raw.loc[raw["source"] == "healthver"]

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
                a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                prediction_raw.append("")

    data_gold = pd.read_json(f"../../dataset/truthfulness/Gold/healthver_gold_2.0.json")

    prediction, gold = [], []
    for p, g in zip(prediction_raw, data_gold["answer"]):
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


def eval_covid(model):
    raw = pd.read_json(f"../../results/truthfulness/{model}/external.json")
    data = raw.loc[raw["source"] == "covid"]

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
                a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                prediction_raw.append("")

    data_gold = pd.read_json(f"../../dataset/truthfulness/Gold/covid_gold_2.0.json")

    prediction, gold = [], []
    for p, g in zip(prediction_raw, data_gold["answer"]):
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


def eval_scifact(model):
    raw = pd.read_json(f"../../results/truthfulness/{model}/external.json")
    data = raw.loc[raw["source"] == "scifact"]

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
                a = i.split("Answer: ", 1)[1]
                prediction_raw.append(a)
            except:
                prediction_raw.append("")

    data_gold = pd.read_json(f"../../dataset/truthfulness/Gold/scifact_gold_2.0.json")

    prediction, gold = [], []
    for p, g in zip(prediction_raw, data_gold["answer"]):
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


def run(results_save_path):
    model_list = []
    for root, dirs, files in os.walk("../../results/truthfulness", topdown=False):
        for name in dirs:
            model_list.append(name)

    result_climate, result_healthver, result_covid, result_scifact = [], [], [], []
    for model in model_list:
        result_climate.append(eval_climate(model))
        result_healthver.append(eval_healthver(model))
        result_covid.append(eval_covid(model))
        result_scifact.append(eval_scifact(model))

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
        f"Finished Evaluating! Results are saved in {os.path.join(results_save_path, 'external.csv')}"
    )
