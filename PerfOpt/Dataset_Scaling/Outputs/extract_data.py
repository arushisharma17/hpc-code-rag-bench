import csv
import json

def format_options(options):
    """
    Format the options dictionary or list to the desired string format. Used to design prompt for evaluation.
    """
    formatted_options = []
    if isinstance(options, dict):
        for key, value in options.items():
            formatted_option = f"{key}. {value}"
            formatted_options.append(formatted_option)
    elif isinstance(options, list):
        for idx, value in enumerate(options, 1):  # Start indexing from 1
            formatted_option = f"{chr(64+idx)}. {value}"  # Convert index to uppercase letter (A, B, C, ...)
            formatted_options.append(formatted_option)
    else:
        raise ValueError("Options are neither a list nor a dictionary.")

    return ', '.join(formatted_options)


def format_and_split_options(options):
    """
    Formats the options dictionary or list into a string and returns individual options separately.
    """
    formatted_options = []
    split_options = {}
    
    if not options:
        return None  # No options provided

    if isinstance(options, dict):
        for key, value in options.items():
            formatted_option = f"{key}. {value}"
            formatted_options.append(formatted_option)
            split_options[key] = value
    elif isinstance(options, list):
        for idx, value in enumerate(options, 1):  # Start indexing from 1
            key = chr(64+idx)  # Convert index to uppercase letter (A, B, C, ...)
            formatted_option = f"{key}. {value}"
            formatted_options.append(formatted_option)
            split_options[key] = value
    else:
        raise ValueError("Options are neither a list nor a dictionary.")

    formatted_options_str = ', '.join(formatted_options)
    print(f"Formatted options: {formatted_options_str}")
    return split_options

def process_json_list(json_list):
    # Processes a list of JSON objects
    entries = []
    for item in json_list:
        question = item.get('Question')
        answer = item.get('Answer')
        # Assume options might not be present for open-ended questions
        options = item.get('Options', None)
        entry = {'Question': question, 'Answer': answer}
        if options:
            entry['Options'] = options
        entries.append(entry)
    return entries

def process_json_dict(json_dict):
    # Processes a dictionary of numbered JSON objects
    entries = []
    for key, value in json_dict.items():
        question = value.get('Question')
        answer = value.get('Answer')
        # Assume options might not be present for open-ended questions
        options = value.get('Options', None)
        entry = {'Question': question, 'Answer': answer}
        if options:
            entry['Options'] = options
        entries.append(entry)
    return entries


def extract_scaled_data(csv_filename):
    evaluation_data = []

    try:
        with open(csv_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                response_data = row['response']
                code_snippet = row['code_snippet']

                try:
                    response_json = json.loads(response_data)

                    if isinstance(response_json, list):
                        entries = process_json_list(response_json)
                    elif isinstance(response_json, dict):
                        entries = process_json_dict(response_json)
                    else:
                        print(f"Unknown JSON format in row: {row}")
                        continue

                    for entry in entries:
                        # Start constructing the evaluation entry
                        evaluation_entry = {
                            'Source': row['source'],
                            'Code Snippet': code_snippet,
                            'Question': entry['Question'],
                            'Answer': entry['Answer']
                        }

                        # Check for options and handle accordingly
                        if 'Options' in entry:
                            if isinstance(entry['Options'], list):
                                # Handle yes/no or multiple-choice options
                                if all(opt in ['Yes', 'No'] for opt in entry['Options']):
                                    # Yes/No options
                                    evaluation_entry.update({'A': 'Yes', 'B': 'No'})
                                else:
                                    # Multiple-choice options
                                    split_options = format_and_split_options(entry['Options'])
                                    evaluation_entry.update(split_options)
                            elif isinstance(entry['Options'], dict):
                                # Multiple-choice options specified as a dictionary
                                split_options = format_and_split_options(entry['Options'])
                                evaluation_entry.update(split_options)
                        else:
                            # Open-ended questions without options
                            evaluation_entry.update({'A': None, 'B': None, 'C': None, 'D': None})

                        evaluation_data.append(evaluation_entry)

                except json.JSONDecodeError:
                    print(f"Error decoding JSON in row: {row} of file: {csv_filename}")

        return evaluation_data
    except FileNotFoundError:
        print(f"File {csv_filename} not found!")
        return []


def main():
    """
    Main function to process multiple CSV files.
    """
    # Path to csv files
    path = "."

    # Prompt the user to enter the numbers of the CSV files to process
    csv_numbers = input("Enter the numbers of the CSV files you want to process (e.g. 0,1,2): ")
    selected_csv_numbers = [int(num.strip()) for num in csv_numbers.split(",")]

    for i in selected_csv_numbers:
        if 0 <= i <= 5:  # Valid range
            csv_filename = f'{path}/rodinia-generated-final_prompt_{i}.csv'
            print(f"Processing file: {csv_filename} ...")

            # Extracting the evaluation data from CSV
            evaluation_results = extract_scaled_data(csv_filename)

            # Only proceed if there is data
            if evaluation_results:
                # Determine if any entry has options to define fieldnames accordingly
                has_options = any('A' in entry for entry in evaluation_results)
                # Define the basic fieldnames (excluding 'Answer')
                fieldnames = ['Source', 'Code Snippet', 'Question']
                # Add option fieldnames if options are present
                if has_options:
                    fieldnames.extend(['A', 'B', 'C', 'D'])
                # Finally, append 'Answer' to be the last field
                fieldnames.append('Answer')

                # Saving the extracted evaluation data to a new CSV
                output_filename = f'{path}/processed_rodinia_{i}.csv'
                with open(output_filename, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for eval_data in evaluation_results:
                        # Make sure each dictionary has the correct keys
                        row = {field: eval_data.get(field, '') for field in fieldnames}
                        writer.writerow(row)

                print(f"Processed data saved to {output_filename}")
            else:
                print(f"No data to process for file: {csv_filename}")

        else:
            print(f"Skipping invalid number: {i}")

if __name__ == "__main__":
    main()



'''
def main():
    """
    Main function to process multiple CSV files.
    """
    # Path to csv files
    path = "."

    # Prompt the user to enter the numbers of the CSV files to process
    csv_numbers = input("Enter the numbers of the CSV files you want to process (e.g. 0,1,2)[0,1: basic and advanced MCQA, 2,3: basic and advanced yes/no, 4,5: basic and advanced open-ended(text)]: ")
    selected_csv_numbers = [int(num.strip()) for num in csv_numbers.split(",")]

    for i in selected_csv_numbers:
        if 0 <= i <= 5:  # Valid range
            csv_filename = f'{path}/rodinia-generated-final_prompt_{i}.csv'
            print(f"Processing file: {csv_filename} ...")

            # Extracting the evaluation data from CSV
            evaluation_results = extract_scaled_data(csv_filename)

            # Displaying the extracted evaluation data
            for eval_data in evaluation_results:
                print(eval_data['prompt'])
                print("Answer:", eval_data['answer'])
                print("-" * 50)

            #save in csv format and put on huggingface
            #load from huggingface and standardize previous manually generated examples as well. Columns - source,code_snippet, question, options, answer. Design prompt as <code_snippet + questions + options>
            #evaluation output: Columns: source, code_snippet + question + options, response, correct_answer, is_correct

            #create different types of prompts and feed to appropriate evaluation function. Import file containing different evaluation functions. standardize the input format of thos functions. 

        else:
            print(f"Skipping invalid number: {i}")

if __name__ == "__main__":
    main()

'''
