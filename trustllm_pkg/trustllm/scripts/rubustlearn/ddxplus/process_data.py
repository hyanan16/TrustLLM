import csv
import json

# Define the input and output file paths
input_file_path = '/Users/lxe/PycharmProjects/robustlearn/chatgpt-robust/data/ddxplus/ddxplus.csv.xls'
output_file_path = '/Users/lxe/PycharmProjects/robustlearn/chatgpt-robust/data/ddxplus/ddxplus.json'

# Read the CSV and write to the JSON file
data_to_convert = []

with open(input_file_path, mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        prompt = f"Given is a patient's information and dialog with the doctor. {row['Information']}. What is the diagnosis? Select one answer among {row['Diag_Set']}."
        output = row['Diagnosis'].lower()
        data_to_convert.append({"prompt": prompt, "output": output})

# Write the JSON output
with open(output_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(data_to_convert, jsonfile, ensure_ascii=False, indent=4)

print("Conversion complete. The JSON file has been saved.")
