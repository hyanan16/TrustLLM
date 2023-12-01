import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize_toxicity(file_path, savepath):
    file_list = os.listdir(file_path)
    print('Totally file number', len(file_list))
    toxicity_data = {}
    line_styles = ['-', '--', '-.', ':', '-']
    colors = sns.color_palette("cubehelix", len(file_list))
    for i, filename in enumerate(file_list):
        assert filename.endswith('.csv')
        print('Processing ', filename)
        toxicity_values = []
        data = pd.read_csv(os.path.join(file_path, filename))
        model_name = os.path.splitext(filename)[0]
        model_name = model_name.replace('_toxic', '')
        model_name = model_name.replace("chatglm", "ChatGLM")
        model_name = model_name.replace("vicuna", "Vicuna")
        model_name = model_name.replace("koala", "Koala")
        model_name = model_name.replace("chatgpt", "ChatGPT")
        model_name = model_name.replace("oasst", "Oasst")
        model_name = model_name.replace("llama", "Llama")
        model_name = model_name.replace('wizardlm', 'WizardLM')
        model_name = model_name.replace('ernie', 'ERNIE')
        model_name = model_name.replace('baichuan', 'Baichuan')
        model_name = model_name.replace('gpt-4', 'GPT-4')
        model_name = model_name.replace('mistral', 'Mistral')
        model_name = model_name.replace('bison-001', 'PaLM 2')
        for index, row in data.iterrows():
            if row['type'] == 'toxicity':
                toxicity_value = row['toxicity']
                if toxicity_value != 'error':
                    toxicity_values.append(toxicity_value)
        toxicity_data[model_name] = toxicity_values

    plt.figure(figsize=(12, 7))
    for i, (model, data) in enumerate(toxicity_data.items()):
        sns.distplot(data, hist=False, label=model, kde_kws={'linestyle': line_styles[i % 5], 'linewidth': 2},
                     color=colors[i])
    plt.xlim(0, 1)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel('Toxicity Value', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.legend(fontsize=13)
    plt.savefig(os.path.join(savepath, 'toxicity_distribution.pdf'), dpi=250)
    plt.show()

    # plt.figure(figsize=(12, 7))
    # model_names = list(toxicity_data.keys())
    # data_values = list(toxicity_data.values())
    # sns.stripplot(data=data_values, palette=colors[:len(model_names)], alpha=0.6)
    # plt.xticks(np.arange(len(model_names)), model_names, rotation=25, fontsize=12)
    # plt.yticks(fontsize=18)
    # plt.xlabel('Model', fontsize=12)
    # plt.ylabel('Toxicity Value', fontsize=18)
    # plt.savefig(os.path.join(savepath, 'toxicity_distribution_scatter.pdf'), dpi=250)
    # plt.show()

    plt.figure(figsize=(12, 7))
    sns.boxplot(data=list(toxicity_data.values()), showfliers=False, palette=colors)
    plt.xticks(range(len(toxicity_data)), list(toxicity_data.keys()), rotation=25, fontsize=12)
    plt.xlabel('Model')
    plt.ylabel('Value', fontsize=18)
    # plt.title('Toxicity Scores Distribution')
    plt.savefig(os.path.join(savepath, 'toxicity_distribution_box.pdf'), dpi=250)
    plt.show()


def calculate_metrics(dir_path, save_path):
    average_values = []
    maximum_values = []
    model_values = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    df = df.dropna(subset=["toxicity"])
                    df["toxicity"] = pd.to_numeric(df["toxicity"], errors='coerce')
                    average = df["toxicity"].mean()
                    maximum = df["toxicity"].max()
                    model_value = df["Model"].iloc[0]
                    model_values.append(model_value)
                    average_values.append(average)
                    maximum_values.append(maximum)
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
    result_df = pd.DataFrame({
        "Model": model_values,
        "average": average_values,
        "maximum": maximum_values
    })

    result_df.to_csv(os.path.join(save_path, 'toxicity_metrics.csv'), index=False)


if __name__ == '__main__':
    visualize_toxicity('../../../results/safety/toxicity/case_toxicity', '../../../assets')
    calculate_metrics('../../../results/safety/toxicity', '../../../results/')
