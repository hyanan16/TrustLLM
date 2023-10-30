import os.path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_robustness_advinstruction(file_path, save_path):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    df_drop_from_one = df.copy()
    perturbation_columns = df.columns[1:-1]
    df_drop_from_one[perturbation_columns] = 1 - df[perturbation_columns]

    # Show the DataFrame with the drop in similarity from 1
    df_drop_from_one.head()
    plt.figure(figsize=(16, 10))

    # Draw a heatmap with the numeric values in each cell
    sns.heatmap(df_drop_from_one.drop(['Average'], axis=1).set_index('Model'), annot=True, fmt=".4f", cmap="coolwarm",
                center=0)
    plt.title('Drop in Embedding Similarity')
    plt.xlabel('Type of Perturbation')
    plt.ylabel('Model')
    plt.savefig(os.path.join(save_path, 'robustness_heatmap_advinstruction.pdf'), dpi=250)
    plt.show()
    avg_drop_per_perturbation = df_drop_from_one[perturbation_columns].mean()

    # Plot the average drop in similarity for each type of perturbation
    plt.figure(figsize=(14, 8))
    sns.barplot(x=avg_drop_per_perturbation.index, y=avg_drop_per_perturbation.values, palette="coolwarm")
    plt.xticks(rotation=45, ha='right')
    # plt.title('Average Drop in Embedding Similarity')
    plt.ylabel('Average Drop in Similarity')
    plt.xlabel('Type of Perturbation')
    plt.savefig(os.path.join(save_path, 'robustness_barplot_advinstruction.pdf'), dpi=250)
    plt.show()


if __name__ == "__main__":
    visualize_robustness_advinstruction('robustness.xlsx', './')