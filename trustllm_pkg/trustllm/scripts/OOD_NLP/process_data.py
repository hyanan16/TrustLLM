import csv
import json

# Define the input TSV file path and the output JSON file path
input_tsv_path = '/Users/lxe/PycharmProjects/OOD_NLP/datasets/process/ToxicDetection/civil_comments/train.tsv'
output_json_path = '/Users/lxe/PycharmProjects/OOD_NLP/datasets/process/ToxicDetection/civil_comments/train.json'

# Create a list to hold the converted data
json_data = []

# Open the TSV file and read rows
with open(input_tsv_path, 'r', encoding='utf-8') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in reader:
        # Convert each row into the desired JSON format
        json_data.append({
            'prompt': row['Text'],
            'label': str(row['Label'])  # Ensure the label is a string
        })

# Write the JSON data to a file
with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(json_data, jsonfile, indent=4)

print(f'Data converted and saved to {output_json_path}')
