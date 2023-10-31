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


if __name__ == "__main__":
    visualize_robustness_advinstruction('../../../results/advinstruction.xlsx', '../../../assets')