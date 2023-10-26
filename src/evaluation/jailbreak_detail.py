import os
import pandas as pd
import json

# 获取模型目录列表
model_directories = [
    "baichuan-13b", "chatglm2", "chatgpt", "ernie", "koala-13b", "llama2-7b",
    "llama2-13b", "oasst-12b", "vicuna-7b", "vicuna-13b", "vicuna-33b", "wizardlm-13b"
]

# 遍历每个模型目录
for model_dir in model_directories:
    model_name = model_dir

    # 初始化一个空的DataFrame
    columns = ["Model", "res", "label", "type"]
    model_df = pd.DataFrame(columns=columns)

    # 遍历 "jailbreak" 和 "toxicity" 两个子文件夹
    for subfolder in ["jailbreak", "toxicity"]:
        subfolder_name = subfolder

        # 构建子文件夹路径
        subfolder_path = os.path.join(model_dir, subfolder)

        # 遍历 JSON 文件
        for json_file in os.listdir(subfolder_path):
            if json_file.endswith(".json") and json_file != "arithmetical.json":
                json_file_path = os.path.join(subfolder_path, json_file)

                # 读取 JSON 文件
                with open(json_file_path, "r") as json_file:
                    data = json.load(json_file)

                    # 遍历 JSON 文件中的每个条目
                    for item in data:
                        longformer_res = item.get("longformer_res", [])
                        for result in longformer_res:
                            res = result.get("label", "")
                            label = item.get("label", [])
                            type_data = subfolder

                            # 如果 "longformer_res" 中有 "LABEL_1"，则添加到临时DataFrame
                            if "LABEL_1" in res:
                                data_to_append = {
                                    "Model": model_name,
                                    "res": item.get("res", ""),
                                    "label": ", ".join(label),
                                    "type": type_data
                                }
                                model_df = pd.concat([model_df, pd.DataFrame([data_to_append])], ignore_index=True)
    print(model_df)
    # 将该模型的数据保存到一个单独的 CSV 文件
    model_df.to_csv(f"{model_name}_output_data.csv", index=False)
