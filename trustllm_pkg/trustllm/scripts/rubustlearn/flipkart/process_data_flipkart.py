import csv
import json

# Define the CSV file path
csv_file_path = '/Users/lxe/PycharmProjects/robustlearn/chatgpt-robust/data/flipkart/Dataset-SA.csv'
# Define the JSON file path
json_file_path = '/Users/lxe/PycharmProjects/robustlearn/chatgpt-robust/data/flipkart/flipkart.json'

# Function to create the prompt string
def create_prompt(rate, review, summary):
    return f"Is the following product review positive, neutral, or negative? Answer with \"positive\", \"neutral\", or \"negative\". Rating:{rate}. Review: {review}. {summary}"

# Read the CSV and write to JSON
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.DictReader(csv_file)
    
    # Create a list to hold JSON entries
    json_data = []
    
    # Iterate over CSV rows
    for row in csv_reader:
        # Construct the prompt
        prompt = create_prompt(row['Rate'], row['Review'], row['Summary'])
        # Construct the JSON entry
        json_entry = {
            'prompt': prompt,
            'output': row['Sentiment']
        }
        # Add the entry to our list
        json_data.append(json_entry)

    # Write the JSON data to file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)

print(f'Data has been converted to JSON and saved to {json_file_path}')
