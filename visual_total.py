# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.rcParams['font.sans-serif'] = ['Athelas']
# # Data for the bar chart
# models = ["ChatGPT", "GPT-4", "ERNIE", "PaLM 2", "Baichuan-13b", "ChatGLM2", "Llama2-7b", "Llama2-13b",
#           "Llama2-70b", "Mistral-7b", "Oasst-12b", "Koala-13b", "Vicuna-7b", "Vicuna-13b", "Vicuna-33b", "Wizardlm-13b"]
# scores = [5.87, 4.84, 6.77, 10.06, 11.61, 9.61, 7.29, 6.03, 5.26, 7.81, 11.61, 11.06, 11.19, 8.03, 7.65, 7.45]
#
# # Sorting the data by scores in ascending order
# ascending_sorted_data = sorted(zip(scores, models))
# ascending_sorted_scores, ascending_sorted_models = zip(*ascending_sorted_data)
#
# # Creating a unique blue color palette for each model
# blue_colors = sns.color_palette("Blues", len(ascending_sorted_models))
#
# # Creating the bar chart with unique blue colors for each model
# plt.figure(figsize=(15, 4))
# sns.barplot(x=list(ascending_sorted_models), y=list(ascending_sorted_scores), palette=blue_colors)
# plt.xlabel('Model', fontsize=15)
# plt.ylabel('Average Ranking', fontsize=15)
# # plt.title('Comparison of Different Models (Ascending Order)')
# plt.xticks(rotation=30, fontsize=15)
# plt.yticks(fontsize=15)
# plt.subplots_adjust(bottom=0.32, top=0.9)
# plt.savefig('assets/avg_rank.pdf', dpi=200)
# plt.show()
