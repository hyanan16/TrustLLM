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
    colors = sns.color_palette("coolwarm", len(file_list))
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
        for index, row in data.iterrows():
            if row['type'] == 'toxicity':
                toxicity_value = row['toxicity']
                if toxicity_value != 'error':
                    toxicity_values.append(toxicity_value)
        toxicity_data[model_name] = toxicity_values

    plt.figure(figsize=(12, 7))
    for i, (model, data) in enumerate(toxicity_data.items()):
        sns.distplot(data, hist=False, label=model, kde_kws={'linestyle': line_styles[i%5], 'linewidth':4}, color=colors[i])
    plt.xlim(0,1)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel('Toxicity Value', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.legend(fontsize=16)
    plt.savefig(os.path.join(savepath, 'toxicity_distribution.pdf'), dpi=250)
    plt.show()

    plt.figure(figsize=(12, 7))
    model_names = list(toxicity_data.keys())
    data_values = list(toxicity_data.values())
    sns.stripplot(data=data_values, palette=colors[:len(model_names)], alpha=0.6)
    plt.xticks(np.arange(len(model_names)), model_names, rotation=25, fontsize=12)
    plt.yticks(fontsize=18)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Toxicity Value', fontsize=18)
    plt.savefig(os.path.join(savepath, 'toxicity_distribution_scatter.pdf'), dpi=250)
    plt.show()


def calculate_metrics(dir_path, save_path):
    pass


if __name__ == '__main__':
    visualize_toxicity('../../../results/safety/toxicity/case_toxicity', '../../../assets')