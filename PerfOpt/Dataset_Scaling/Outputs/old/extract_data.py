import csv
import json

# Open the CSV file
with open('rodinia-generated-questions_prompt_2.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    
    # Iterate over rows in CSV
    for row in reader:
        response_data = row['response']
        
        # Parse the JSON data in 'responses' column
        try:
            response_list = json.loads("[" + response_data + "]")  # Wrapping in [] to create a JSON array
            for resp in response_list:
                question = resp.get('Question')
                answer = resp.get('Answer')
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print("-" * 50)  # Just a separator
        except json.JSONDecodeError:
            print("Error decoding JSON in row:", row)

