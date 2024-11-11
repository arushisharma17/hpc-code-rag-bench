import csv
import json

def format_options(options_dict):
    """
    Format the options dictionary to the desired string format. Used to design prompt for evaluation. 
    """
    formatted_options = []
    for key, value in options_dict.items():
        formatted_option = f"{key}. {value}"
        print("formatted_option", formatted_option)
        formatted_options.append(formatted_option)
    
    return ', '.join(formatted_options)

def extract_scaled_data(csv_filename):
    """
    Extracts and prints formatted data from the given CSV filename.
    """
    formatted_data = []
    evaluation_data = []  # To store data formatted for LLM QA evaluation: list of dictionaries - prompt, answer.

    try:
        with open(csv_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            # Iterate over rows in CSV
            for row in reader:
                response_data = row['response']
                code_snippet = row['code_snippet']

                # Parse the JSON data in 'responses' column
                try:
                    response_list = json.loads(response_data)

                    for resp in response_list:
                        entry = {}

                        question = resp.get('Question')
                        answer = resp.get('Answer')
                        options = resp.get('Options')

                        if question and options and answer:  # Checking if all fields are present
                            entry['Question'] = question
                            entry['Options'] = options
                            entry['Answer'] = answer

                            formatted_data.append(entry)

                            # Create an evaluation entry

                            formatted_options_str = format_options(options)
                            evaluation_entry = {
                                    'prompt': f"{code_snippet}\n{question}\nOptions: {formatted_options_str}",
                                    'answer': answer
                                    }
                            evaluation_data.append(evaluation_entry)

                except json.JSONDecodeError:
                    print(f"Error decoding JSON in row: {row} of file: {csv_filename}")

        # Printing the extracted data in the desired format
        print(json.dumps(formatted_data, indent=4))

        # Return evaluation data
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

