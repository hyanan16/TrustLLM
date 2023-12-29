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


def eval_halu_dial(folderpath, model):
    raw = pd.read_json(f"{folderpath}/truthfulness/{model}/hallucination.json")
    data = raw.loc[raw["source"] == "halu_dial"]

    original = pd.read_json("../../dataset/truthfulness/hallucination.json").loc[
            raw["source"] == "halu_dial"
    ]

    pred = []
    for i in data["res"]:
        if i is not None:
            if "yes" in i.lower():
                pred.append("Yes")
            elif "no" in i.lower():
                pred.append("No")
            else:
                pred.append("NA")
        else:
            pred.append("NA")

    gold = original["answer"]

    assert len(pred) == len(gold) == len(data)

    correct = 0
    for p, g in zip(pred, gold):
        if p == g:
            correct = correct + 1
    return correct / len(data)


def eval_halu_qa(folderpath, model):
    raw = pd.read_json(f"{folderpath}/truthfulness/{model}/hallucination.json")
    data = raw.loc[raw["source"] == "halu_qa"]

    original = pd.read_json("../../dataset/truthfulness/hallucination.json").loc[
        raw["source"] == "halu_qa"
    ]

    pred = []
    for i in data["res"]:
        try:
            if "yes" in i.lower():
                pred.append("Yes")
            elif "no" in i.lower():
                pred.append("No")
            else:
                pred.append("NA")
        except:
            pred.append("NA")

    gold = original["answer"]

    assert len(pred) == len(gold) == len(data)

    correct = 0
    for p, g in zip(pred, gold):
        if p == g:
            correct = correct + 1
    return correct / len(data)


def eval_halu_summ(folderpath, model):
    raw = pd.read_json(f"{folderpath}/truthfulness/{model}/hallucination.json")
    data = raw.loc[raw["source"] == "halu_summ"]

    original = pd.read_json("../../dataset/truthfulness/hallucination.json").loc[
        raw["source"] == "halu_summ"
    ]

    pred = []
    for i in data["res"]:
        try:
            if "yes" in i.lower():
                pred.append("Yes")
            elif "no" in i.lower():
                pred.append("No")
            else:
                pred.append("NA")
        except:
            pred.append("NA")

    gold = original["answer"]

    assert len(pred) == len(gold) == len(data)

    correct = 0
    for p, g in zip(pred, gold):
        if p == g:
            correct = correct + 1
    return correct / len(data)


def eval_mc(folderpath, model):
    raw = pd.read_json(f"{folderpath}/truthfulness/{model}/hallucination.json")
    data = raw.loc[raw["source"] == "mc"]

    pred = []
    for i in data["res"]:
        if model == "llama2-13b":
            if len(i.split("\n", 1)) == 1:
                for j in i.split("\n", 1)[0]:
                    if j in ["A.", "A", "B.", "B", "C.", "C", "D.", "D"]:
                        pred.append(j)
            else:
                for j in i.split("\n", 1)[1]:
                    if j in ["A.", "A", "B.", "B", "C.", "C", "D.", "D"]:
                        pred.append(j)
        elif model == "ernie":
            for j in i:
                if j in ["A.", "A", "B.", "B", "C.", "C", "D.", "D"]:
                    pred.append(j)
        elif model == "baichuan-13b":
            for j in i.split(":", 1):
                if j in ["A", "B", "C", "D"]:
                    pred.append(j)
        else:
            pred.append(re.sub(r"[^\w\s]", "", i.split()[0]))

    for i in range(len(pred)):
        if pred[i] not in ["A", "B", "C", "D"]:
            pred[i] = "NA"

    correct = 0
    for i in pred:
        if i == "A":
            correct = correct + 1
    return round(correct / len(data), 3)


def run(folderpath, results_save_path):
    model_list = []
    for root, dirs, files in os.walk("../../results/truthfulness", topdown=False):
        for name in dirs:
            model_list.append(name)

    result_dial, result_qa, result_sum, result_mc = [], [], [], []
    for model in model_list:
        print(f"\033[93mEvaluating {model} ... ...\033[0m")
        result_dial.append(eval_halu_dial(folderpath, model))
        result_qa.append(eval_halu_qa(folderpath, model))
        result_sum.append(eval_halu_summ(folderpath, model))
        result_mc.append(eval_mc(folderpath, model))

    assert len(result_dial) == len(result_qa) == len(result_sum) == len(result_mc)

    result_df = pd.DataFrame(
        list(zip(model_list, result_dial, result_qa, result_sum, result_mc)),
        columns=["model", "KGD", "QA", "SUM", "MC"],
    )
    result_df.to_csv(os.path.join(results_save_path, "hallucination.csv"), index=False)
    print(
        f"\033[92mFinished Evaluating! Results are saved in {os.path.join(results_save_path, 'hallucination.csv')}\033[0m"
    )
