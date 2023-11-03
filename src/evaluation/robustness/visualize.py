import os.path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def visualize_robustness_advinstruction(file_path, save_path):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    df_drop_from_one = df.copy()
    perturbation_columns = df.columns[1:-1]
    df_drop_from_one[perturbation_columns] = 1 - df[perturbation_columns]

    # Set the font scale for better readability
    sns.set(font_scale=1.2)

    # Show the DataFrame with the drop in similarity from 1
    df_drop_from_one.head()
    plt.figure(figsize=(20, 12))  # Adjusted figure size

    # Draw a heatmap with the numeric values in each cell
    sns.heatmap(df_drop_from_one.drop(['Average'], axis=1).set_index('Model'), annot=True, fmt=".4f", cmap="coolwarm",
                center=0)
    plt.title('Drop in Embedding Similarity', fontsize=20)
    plt.xlabel('Type of Perturbation', fontsize=20)  # Increased font size
    plt.ylabel('Model', fontsize=20)  # Increased font size
    plt.xticks(rotation=25, fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(save_path, 'robustness_heatmap_advinstruction.pdf'), dpi=300)  # Adjusted DPI
    plt.show()
    avg_drop_per_perturbation = df_drop_from_one[perturbation_columns].mean()

    # Plot the average drop in similarity for each type of perturbation
    plt.figure(figsize=(16, 10))  # Adjusted figure size
    sns.barplot(x=avg_drop_per_perturbation.index, y=avg_drop_per_perturbation.values, palette="coolwarm")
    plt.xticks(rotation=15, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title('Average Drop in Embedding Similarity')
    plt.ylabel('Average Drop in Similarity', fontsize=20)  # Increased font size
    # plt.xlabel('Type of Perturbation', fontsize=20)  # Increased font size
    plt.savefig(os.path.join(save_path, 'robustness_barplot_advinstruction.pdf'), dpi=300)  # Adjusted DPI
    plt.show()

# Example usage:
# visualize_robustness_advinstruction('path_to_your_excel_file.xlsx', 'path_to_save_directory')


def visual_total_score(save_path):
    import matplotlib.pyplot as plt

    # Data for LLMs and their corresponding scores
    llm_names = [
        'Baichuan-13b', 'ChatGLM2', 'Vicuna-13b', 'Vicuna-7b', 'Vicuna-33b',
        'Llama2-7b', 'Llama2-13b', 'Koala-13b', 'Oasst-12b', 'WizardLM-13b',
        'ERNIE', 'ChatGPT', 'GPT-4', 'Llama2-70b'
    ]
    llm_scores = [
        0.363, 0.254, 0.180, 0.072, 0.219,
        0.374, 0.306, 0.116, 0.143, 0.152,
        0.408, 0.326, 0.591, 0.471
    ]

    # Pair the names and scores, then sort them by score
    llm_data = zip(llm_names, llm_scores)
    sorted_llm_data = sorted(llm_data, key=lambda x: x[1], reverse=True)

    # Unzip into two lists after sorting
    sorted_names, sorted_scores = zip(*sorted_llm_data)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_names, sorted_scores, color='skyblue')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=11.5)
    plt.xlabel('Scores', fontsize=16)
    plt.title('ACC (adv) - ASR', fontsize=18)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top
    plt.savefig(os.path.join(save_path, 'advglue_score.pdf'), dpi=200)
    plt.show()


if __name__ == "__main__":
    # visualize_robustness_advinstruction('../../../results/advinstruction.xlsx', '../../../assets')
    visual_total_score('../../../assets')