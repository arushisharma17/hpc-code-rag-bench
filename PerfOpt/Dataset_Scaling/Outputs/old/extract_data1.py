import csv
import json

# Open the CSV file
csv_filename = 'rodinia-generated-questions_prompt_0.csv'

questions = []
answers = []

with open(csv_filename, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    
    # Iterate over rows in CSV
    for row in reader:
        response_data = row['response']
        
        # Parse the JSON data in 'responses' column
        try:
            response_list = json.loads("[" + response_data + "]")  # Wrapping in [] to create a JSON array
            
            for resp in response_list:
                if isinstance(resp, dict):
                    question = resp.get('Question')
                    answer = resp.get('Answer')

                    if question and answer:  # Checking if both are not None
                        questions.append(question)
                        answers.append(answer)
                
                elif isinstance(resp, list):  # Handle nested lists
                    for item in resp:
                        question = item.get('Question')
                        answer = item.get('Answer')

                        if question and answer:
                            questions.append(question)
                            answers.append(answer)

        except json.JSONDecodeError:
            print("Error decoding JSON in row:", row)

# Printing the extracted questions and answers for verification
for q, a in zip(questions, answers):
    print(f"Question: {q}")
    print(f"Answer: {a}")
    print("-" * 50)  # Just a separator

