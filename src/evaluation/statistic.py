import os
import pandas as pd
import json

# 获取当前目录下的文件列表
file_list = os.listdir()

# 初始化一个字典来存储不同攻击模式的成功率
attack_success_rates = {}

# 遍历每个模式的文件夹
for attack_mode in ["arithmetical", "bad_words", "cot", "fixed_sentence", "json_format", "leetspeak", "multi_task", "no_long_word_prompt", "no_punctuation", "programming", "Refusal sentence prohibition", "scenario", "url_encode", "without_the"]:
    total_jailbreak_label_1 = 0
    total_toxicity_label_1 = 0
    total_overall_label_1 = 0
    total_jailbreak_prompts = 0
    total_toxicity_prompts = 0
    total_overall_prompts = 0

    # 遍历每个模型文件夹
    for model_dir in ["baichuan-13b", "chatglm2", "chatgpt", "ernie", "koala-13b", "llama2-7b", "llama2-13b", "oasst-12b", "vicuna-7b", "vicuna-13b", "vicuna-33b", "wizardlm-13b"]:
        # 构建子目录路径
        jailbreak_dir = os.path.join(model_dir, "jailbreak")
        toxicity_dir = os.path.join(model_dir, "toxicity")

        # 处理 jailbreak 目录下的 JSON 文件
        jailbreak_json_path = os.path.join(jailbreak_dir, attack_mode + ".json")
        if os.path.exists(jailbreak_json_path):
            with open(jailbreak_json_path, "r") as json_file:
                data = json.load(json_file)
                for item in data:
                    if "label" in item and "longformer_res" in item:
                        if "LABEL_1" in item["longformer_res"][0]["label"]:
                            total_jailbreak_label_1 += 1
                    total_jailbreak_prompts += 1

        # 处理 toxicity 目录下的 JSON 文件
        toxicity_json_path = os.path.join(toxicity_dir, attack_mode + ".json")
        if os.path.exists(toxicity_json_path):
            with open(toxicity_json_path, "r") as json_file:
                data = json.load(json_file)
                for item in data:
                    if "label" in item and "longformer_res" in item:
                        if "LABEL_1" in item["longformer_res"][0]["label"]:
                            total_toxicity_label_1 += 1
                    total_toxicity_prompts += 1

    total_overall_label_1 = total_jailbreak_label_1 + total_toxicity_label_1
    total_overall_prompts = total_jailbreak_prompts + total_toxicity_prompts

    # 计算成功率并存储到字典
    if total_jailbreak_prompts > 0 or total_toxicity_prompts > 0:
        attack_success_rates[attack_mode] = {
            "label": attack_mode,  # 添加 "label" 列
            "jailbreak": total_jailbreak_label_1 / total_jailbreak_prompts,
            "toxicity": total_toxicity_label_1 / total_toxicity_prompts,
            "overall": total_overall_label_1 / total_overall_prompts,
        }

# 创建DataFrame并保存为CSV文件
df = pd.DataFrame(attack_success_rates).transpose()
print(df.to_string(index=False))
df.to_csv("label_success_rates.csv", index=False)

# 第二部分代码
# 获取所有模型文件夹的名称
model_directories = [
    "baichuan-13b", "chatglm2", "chatgpt", "ernie", "koala-13b", "llama2-7b",
    "llama2-13b", "oasst-12b", "vicuna-7b", "vicuna-13b", "vicuna-33b", "wizardlm-13b"
]

# 初始化一个字典来存储每个模型的攻击成功率
model_success_rates = {}

# 创建DataFrame的列名，包括模型名称
columns = ["Model", "jailbreak", "toxicity", "overall"]

# 创建一个空的列表来存储数据
data_list = []

# 遍历每个模型目录
for model_dir in model_directories:
    model_success_rates[model_dir] = {}

    # 遍历模式
    attack_modes = os.listdir(model_dir)

    # 初始化总数
    total_label_1_jailbreak = 0
    total_prompts_jailbreak = 0
    total_label_1_toxicity = 0
    total_prompts_toxicity = 0

    success_jailbreak = 0
    success_toxicity = 0

    for attack_mode in attack_modes:
        attack_mode_dir = os.path.join(model_dir, attack_mode)

        # 统计LABEL_1的总数和所有prompt的总数
        total_label_1 = 0
        total_prompts = 0

        # 遍历json文件
        json_files = os.listdir(attack_mode_dir)
        for json_file in json_files:
            if json_file.endswith(".json"):
                with open(os.path.join(attack_mode_dir, json_file), "r") as f:
                    data = json.load(f)
                    for item in data:
                        labels = item.get("label", [])  # Use a default empty list if "label" is missing
                        for label in labels:
                            total_label_1 += label.count("LABEL_1")

                        longformer_res = item.get("longformer_res", [])
                        for result in longformer_res:
                            if result.get("label") == "LABEL_1":
                                total_label_1 += 1

                        total_prompts += 1

        # 计算总数
        if attack_mode == "jailbreak":
            total_label_1_jailbreak += total_label_1
            total_prompts_jailbreak += total_prompts
            success_jailbreak = total_label_1_jailbreak / total_prompts_jailbreak
        elif attack_mode == "toxicity":
            total_label_1_toxicity += total_label_1
            total_prompts_toxicity += total_prompts
            success_toxicity = total_label_1_toxicity / total_prompts_toxicity

        # 计算 "overall"
        overall = (total_label_1_jailbreak + total_label_1_toxicity) / (
                    total_prompts_jailbreak + total_prompts_toxicity)

    # 添加每个模型的数据到数据列表
    model_data = {
        "Model": model_dir,
        "jailbreak": success_jailbreak,
        "toxicity": success_toxicity,
        "overall": overall
    }
    data_list.append(model_data)

# 创建DataFrame
df = pd.DataFrame(data_list, columns=columns)

# 打印结果
print(df.to_string(index=False))

# 保存为CSV文件
df.to_csv("model_success_rates.csv", index=False)
