'''To scale data using code snippets form original rodinia-benchmark based mcq dataset with standardized prompts found in prompts.py. '''

#from prompts import PROMPTS
import os
from datasets import load_dataset
import argparse
import csv
# Local imports
from lm4hpc.hpcpipeline import hpcpipelines
import sys
from datasets import load_dataset

# Set up external services
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
print(openai.api_key)


def generate_dataset_with_code(dataset, model_name, output_csv_base):

    # Initialize model pipeline
    OMP_QA_sc = hpcpipelines(task="openmp_question_answering", model=model_name, pdf_files="", langchain_embedding="")

    # Define various types of prompts as dictionaries with an index and text
    prompts = [
        {
            "index": 0, 
            "text": "Generate 10 OpenMP performance optimization multiple choice questions based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: Question :<generated question>, Options:<Four options A, B,C and D with one correct answer>, Answer: Correct answer to the generated question 'A', 'B', 'C' or 'D'>"
        },
        {
            "index": 1, 
            "text": "Generate 10  multiple choice questions about advanced OpenMP performance optimization concepts based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: Question :<generated question>, Options:<Four options A, B,C and D with one correct answer>, Answer: Correct answer to the generated question 'A', 'B', 'C' or 'D'>"
        },           
        {
            "index": 2, 
            "text": "Generate 10 OpenMP performance optimization Yes/No questions based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: <generated question> Answer: <Correct answer to the generated question 'Yes' or 'No'>"
        },
        {
            "index": 3, 
            "text": "Generate 10 Yes/No questions about advanced OpenMP performance optimization concepts based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question> Answer: <Correct answer to the generated question 'Yes' or 'No'>"
        },
        {
            "index": 4, 
            "text": "Generate 10 open-ended OpenMP performance optimization questions based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question> Answer: <Correct answer to the generated question>"
        },
        {
            "index": 5, 
            "text": "Generate 10 open-ended questions about advanced OpenMP performance optimization concepts based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question> Answer: <Correct answer to the generated question>"
        }
    ]


    # Selected Rodinia samples
    row_indices = [2, 11, 22, 42, 67, 86, 115, 127 ]

    #prompts stored in PROMPTS in prompts.py
    for prompt in prompts:

        # Construct the CSV filename for this prompt
        output_csv = f"Outputs/{output_csv_base}_prompt_{prompt['index']}.csv"

        # Open the CSV file for writing
        with open(output_csv, "w", newline="") as csvfile:
            fieldnames = ["source","code_snippet", "prompt", "response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for idx in row_indices:
                code_snippet = dataset['train']['Code (optional)'][idx]
                source =  dataset['train']['Source'][idx]
                print(source, code_snippet)
                full_prompt = code_snippet + " " + prompt["text"]
                print("full prompt", full_prompt)
                response = OMP_QA_sc(full_prompt)
                print(response)

                writer.writerow({
                    "source": source,
                    "code_snippet": code_snippet,
                    "prompt": full_prompt,
                    "response": response
                })

    print("CSV generation completed.")

def generate_dataset( model_name, output_csv_base):

    # Initialize model pipeline
    OMP_QA_sc = hpcpipelines(task="openmp_question_answering", model=model_name, pdf_files="", langchain_embedding="")
    
    # Define various types of prompts as dictionaries with an index and text

    prompts = [
    {
        "index": 0,
        "text": "Generate 10 OpenMP performance optimization multiple choice questions. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question>, Options: <Four options A, B, C, and D with one correct answer>, Answer: Correct answer to the generated question 'A', 'B', 'C', or 'D'."
    },
    {
        "index": 1,
        "text": "Generate 10 multiple choice questions about advanced OpenMP performance optimization concepts. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question>, Options: <Four options A, B, C, and D with one correct answer>, Answer: Correct answer to the generated question 'A', 'B', 'C', or 'D'."
    },
    {
        "index": 2,
        "text": "Generate 10 OpenMP performance optimization Yes/No questions. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question>, Answer: <Correct answer to the generated question 'Yes' or 'No'>."
    },
    {
        "index": 3,
        "text": "Generate 10 Yes/No questions about advanced OpenMP performance optimization concepts. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question>, Answer: <Correct answer to the generated question 'Yes' or 'No'>."
    },
    {
        "index": 4,
        "text": "Generate 10 open-ended OpenMP performance optimization questions. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question>, Answer: <Correct answer to the generated question>."
    },
    {
        "index": 5,
        "text": "Generate 10 open-ended questions about advanced OpenMP performance optimization concepts. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question>, Answer: <Correct answer to the generated question>."
    }
]

    for prompt in prompts:

        # Construct the CSV filename for this prompt
        output_csv = f"Outputs/{output_csv_base}_prompt_{prompt['index']}.csv"

        # Open the CSV file for writing
        with open(output_csv, "w", newline="") as csvfile:
            fieldnames = ["prompt", "response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            full_prompt = prompt["text"]  
            response = OMP_QA_sc(full_prompt)
            print(response)

            writer.writerow({
                "prompt": full_prompt,
                "response": response
                })

    print("CSV generation completed.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale Rodinia dataset using standardized prompts.")
    parser.add_argument("--mcqa_dataset_file", nargs='+', default=["mcq-single-orig.csv"], help="Paths to the MCQA dataset files.")
    parser.add_argument("--open_ended_dataset_file", nargs='+', default=["code.csv"], help="Paths to the open-ended dataset files.")
    parser.add_argument("--model_names", nargs='+', default=["gpt-4"], help="List of model names to evaluate.")
    args = parser.parse_args()
   
    
    #Load Rodinia Dataset
    #rodinia_dataset = load_dataset("sharmaarushi17/HPCPerfOpt-MCQA", data_files="rodinia-chatgpt-mcq-orig.csv")
    #print(rodinia_dataset)

    #generate_dataset(rodinia_dataset,"databricks/dolly-v2-3b","rodinia-generated-questions.csv")
    #generate_dataset_with_code(rodinia_dataset,"gpt-4","rodinia-generated-final")

    generate_dataset("gpt-4", "rodinia-general")

    
    #Load ompify dataset
    #ompify_dataset




