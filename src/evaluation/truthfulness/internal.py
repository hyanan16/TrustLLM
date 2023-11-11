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
import openai
from openai.embeddings_utils import cosine_similarity


def eval_codah(model):
    raw = pd.read_json(f"../../results/truthfulness/{model}/internal.json")
    codah = raw.loc[raw["source"] == "codah"]

    prediction = []
    for i in codah["res"]:
        if model == "baichuan-13b":
            prediction.append(i)
        if model == "ernie":
            s = i.split("result")[1].split("is_truncated")[0]
            rs = [int(s) for s in re.findall(r"\b\d+\b", s)]
            if rs == []:
                prediction.append(0)
            else:
                prediction.append(rs[0])
        else:
            try:
                a = i.split("Answer: ", 1)[1]
                if a == "1" or a == "2" or a == "3" or a == "4":
                    prediction.append(a)
                else:
                    temp = re.findall(r"\d+", a)
                    res = list(map(int, temp))
                    if res:
                        prediction.append(res[0])
                    else:
                        prediction.append(0)
            except:
                prediction.append(0)

    codah_gold = pd.read_json(f"../../dataset/truthfulness/Gold/codah_gold_1.0.json")
    gold = codah_gold["answer"]

    assert len(prediction) == len(gold)

    correct = 0
    for p, g in zip(prediction, gold):
        if p == str(g):
            correct = correct + 1
    return correct / len(gold)


def eval_squad(model):
    raw = pd.read_json(f"../../results/truthfulness/{model}/internal.json")
    data = raw.loc[raw["source"] == "squad"]

    prediction = []
    for i in data["res"]:
        try:
            a = i.split("Answer: ", 1)[1]
            prediction.append(a)
        except:
            prediction.append(i)

    data_gold = pd.read_json(f"../../dataset/truthfulness/Gold/squad_gold_2.0.json")
    gold = data_gold["answers"]

    with open("../../dataset/truthfulness/Processed/squad.pkl", "rb") as f:
        question = pickle.load(f)["question"]

    result = []
    for q, p, g in tqdm(
        zip(question, prediction, gold), total=len(prediction), desc="eval squad"
    ):
        prompt = f"Given a question and a list of correct answers. Is the following answer correct? Only reply YES or NO \n Question: {q} \n Correct Answer: {g} \n Answer to rate: {p}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=10,
        )
        result.append(response.choices[0].text)

    num_yes = 0
    num_no = 0
    for i in result:
        if "yes" in i.lower():
            num_yes = num_yes + 1
        if "no" in i.lower():
            num_no = num_no + 1
    return round(num_yes / (num_yes + num_no), 3)


def eval_adv(model):
    raw = pd.read_json(f"../../results/truthfulness/{model}/internal.json")
    data = raw.loc[raw["source"] == "adversarial"]

    prediction = []
    for i in data["res"]:
        try:
            a = i.split("Answer: ", 1)[1]
            prediction.append(a)
        except:
            prediction.append(i)

    data_gold = pd.read_json(
        f"../../dataset/truthfulness/Gold/adversarial_gold_2.0.json"
    )
    gold = data_gold["answer"]

    adv = pd.read_pickle("../../dataset/truthfulness/Processed/adversarial.pkl")

    context, ids, questions, answers = [], [], [], []
    for i in adv:
        context.append(i["paragraphs"][0]["context"])
        ids.append(i["paragraphs"][0]["qas"][0]["id"])
        questions.append(i["paragraphs"][0]["qas"][0]["question"])
        answers.append(i["paragraphs"][0]["qas"][0]["answers"][0]["text"])

    assert len(context) == len(ids) == len(questions) == len(answers)

    result = []
    for q, p, g in tqdm(
        zip(questions, prediction, gold), total=len(prediction), desc="eval adv"
    ):
        prompt = f"Given a question and a list of correct answers. Is the following answer correct? Only reply YES or NO \n Question: {q} \n Correct Answer: {g} \n Answer to rate: {p}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=10,
        )
        result.append(response.choices[0].text)

    num_yes = 0
    num_no = 0
    for i in result:
        if "yes" in i.lower():
            num_yes = num_yes + 1
        if "no" in i.lower():
            num_no = num_no + 1
    return round(num_yes / (num_yes + num_no), 3)


def eval_hotpot(model):
    raw = pd.read_json(f"../../results/truthfulness/{model}/internal.json")
    data = raw.loc[raw["source"] == "hotpot"]

    prediction = []
    for i in data["res"]:
        try:
            a = i.split("Answer: ", 1)[1]
            prediction.append(a)
        except:
            prediction.append(i)

    data_gold = pd.read_json(f"../../dataset/truthfulness/Gold/hotpot_gold_2.0.json")
    gold = data_gold["answers"]

    with open(f"../../dataset/truthfulness/Processed/hotpot.pkl", "rb") as f:
        question = pickle.load(f)["question"]

    result = []
    for q, p, g in tqdm(
        zip(question, prediction, gold), total=len(prediction), desc="eval hotpot"
    ):
        prompt = f"Given a question and a list of correct answers. Is the following answer correct? Only reply YES or NO \n Question: {q} \n Correct Answer: {g} \n Answer to rate: {p}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=10,
        )
        result.append(response.choices[0].text)

    num_yes = 0
    num_no = 0
    for i in result:
        if "yes" in i.lower():
            num_yes = num_yes + 1
        if "no" in i.lower():
            num_no = num_no + 1
    return round(num_yes / (num_yes + num_no), 3)


def run(results_save_path):
    model_list = []
    for root, dirs, files in os.walk("../../results/truthfulness", topdown=False):
        for name in dirs:
            model_list.append(name)

    result_codah, result_squad, result_adv, result_hotpot = [], [], [], []
    for model in model_list:
        print(f"Evaluating {model}... ...")
        result_codah.append(eval_codah(model))
        result_squad.append(eval_squad(model))
        result_adv.append(eval_adv(model))
        result_hotpot.append(eval_hotpot(model))

    assert (
        len(result_codah) == len(result_squad) == len(result_adv) == len(result_hotpot)
    )

    result_df = pd.DataFrame(
        list(zip(model_list, result_codah, result_squad, result_adv, result_hotpot)),
        columns=["model", "CODAH", "SQuAD2.0", "AdversarialQA", "HotpotQA"],
    )
    result_df.to_csv(os.path.join(results_save_path, "internal.csv"), index=False)
    print(
        f"Finished Evaluating! Results are saved in {os.path.join(results_save_path, 'internal.csv')}"
    )
