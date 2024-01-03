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











# import json
# with open('dataset/ethics/explicit_moralchoice_high_ambiguity.json', 'r') as f:
#     data_1 = json.load(f)
#
# with open('dataset/ethics/explicit_moralchoice_low_ambiguity.json', 'r') as f:
#     data_2 = json.load(f)
#
# for el in data_1:
#     el['type'] = 'high'
# for el in data_2:
#     el['type'] = 'low'
#
# data = data_1 + data_2
#
# with open('dataset/ethics/explicit_moralchoice.json', 'w') as f:
#     json.dump(data, f, indent=4)

import pandas as pd

# Load the Excel file
file_path = 'authorlist.csv'
data = pd.read_csv(file_path)

all_data = []
institution_list = []

for index, el in data.iterrows():
    if el['Author'] and el['Institution']:
        if el['Institution'] not in institution_list:
            institution_list.append(el['Institution'])
            number = len(el['Institution'])
        if el['Institution'] in institution_list:
            number = institution_list.index(el['Institution']) + 1
        all_data.append({'author': el['Author'], 'institution': el['Institution'], 'number': number})

total = 0
number_count = 0
corresponding = ['Yue Huang', 'Lichao Sun']
major = ['Yue Huang', 'Lichao Sun', 'Siyuan Wu', 'Haoran Wang', 'Yixin Huang', 'Yixin Liu',
         'Zhengliang Liu', 'Wenhan Lyu', 'Yixuan Zhang', 'Chujie Gao', 'Qihui Zhang', 'Xiner Li', 'Yijue Wang', 'Zhikun Zhang']
visiting = ['Yue Huang', 'Siyuan Wu', 'Qihui Zhang', 'Chujie Gao']
for index, el in enumerate(all_data):
    total += len(el['author'])
    number += 1
    # 每六个添加一个换行符
    cut = '\quad'
    end = ''
    if number_count == 6 or total > 50:
        cut = '\\\\'
        total = 0
        number_count = 0
    if el['author'] in major:
        end += '\\footnotemark[1]'
    if el['author'] in corresponding:
        end += '~~\\footnotemark[2]'

    if el['author'] in visiting:
        end += '~~\\footnotemark[3]'
    print("{\\bfseries " + el['author'] + '$^{' + str(el['number']) + '}$' + end + '}' + cut)

printed_numbers = set()

# 使用sorted函数按照'number'键进行排序
sorted_data = sorted(all_data, key=lambda x: x['number'])
print('\\\\')


number_count = 0

# 打印排序后的'institution'，但不重复打印相同 'number' 的 'institution'
for item in sorted_data:
    number = item['number']
    institution = item['institution']

    cut = '\\quad'
    if number not in printed_numbers:
        number_count += len(item['institution'])
        if number_count > 55:
            cut = '\\\\'
            number_count = 0
        print("{\\bfseries $^{" + str(item['number']) + "}$" + institution.replace('&', '\\&') + '}' + cut)

        # 将 'number' 添加到已打印的集合中
        printed_numbers.add(number)