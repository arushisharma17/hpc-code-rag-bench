import csv
import json

# Path to the input CSV file
input_csv_file = 'input.csv'

# Path to the output CSV file
output_csv_file = 'questions_extracted.csv'

# Read the CSV file and parse the JSON in the 'response' column
with open(input_csv_file, newline='') as file:
    reader = csv.DictReader(file)
    questions_data = []
    for row in reader:
        # Assuming the JSON is in the 'response' column
        response_json = row['response']
        try:
            data = json.loads(response_json)
            if 'Questions' in data:
                questions_data.extend(data['Questions'])
        except json.JSONDecodeError:
            print(f"Invalid JSON in row: {row}")

# Write the parsed questions to a new CSV file
with open(output_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Answer'])

    for question in questions_data:
        writer.writerow([
            question['Question'],
            question['Options']['A'],
            question['Options']['B'],
            question['Options']['C'],
            question['Options']['D'],
            question['Answer']
        ])

print(f"Questions extracted to {output_csv_file}")

