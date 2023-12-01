import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def jailbreak_type_vis_radar(file_path='../../../results/jailbreak_type_stat.csv', save_path='../../../assets/different_label_type.pdf'):
    data = pd.read_csv(file_path)
    labels = data['label']
    overall_values = data['overall'].tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    overall_values += overall_values[:1]  # Complete the loop for data
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, overall_values, color='r', linewidth=2, linestyle='solid')
    ax.fill(angles, overall_values, color='r', alpha=0.4)
    ax.set_title('Overall Effectiveness', size=16, color='g', y=1.1)
    ax.set_ylim(0, 0.6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=14, color='black')
    ax.set_rlabel_position(0)
    ax.set_yticks([0.1, 0.3, 0.5])
    ax.set_yticklabels(["0.1", "0.3", "0.5"], color="black", size=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def jailbreak_type_vis_bar(file_path='../../../results/jailbreak_type_stat.csv', save_path='../../../assets/different_label_type.pdf'):
    data = pd.read_csv(file_path)
    labels = data['label']
    overall_values = data['overall'].tolist()
    llm_data = zip(labels, overall_values)
    sorted_llm_data = sorted(llm_data, key=lambda x: x[1], reverse=True)

    # Unzip into two lists after sorting
    sorted_names, sorted_scores = zip(*sorted_llm_data)

    # Define the coolwarm colormap
    cmap = plt.get_cmap('inferno')
    colors = [cmap(i / len(sorted_names)) for i in range(len(sorted_names))]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_names, sorted_scores, color=colors)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13.5)
    plt.xlabel('1 - RtA', fontsize=16)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top

    plt.subplots_adjust(left=0.3, right=0.7)
    plt.savefig(os.path.join(save_path), dpi=200)
    plt.show()


jailbreak_type_vis_bar()