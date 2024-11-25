from prompts import EVALUATION_PROMPTS
from datasets import load_dataset
from pipeline import hpcpipelines
from sentence_transformers import SentenceTransformer, util
import argparse
import csv
import os
import json
import re
import openai
from nltk.translate.bleu_score import sentence_bleu
import code_bert_score


def load_dataset_from_hub(dataset_type, data_file, test_mode):
    """
    Load a dataset from the Hugging Face Hub based on the dataset type and file provided.

    Parameters:
    - dataset_type (str): The type of dataset to load. Accepted values are "mcq" for multiple-choice questions
                          and "open_ended" for open-ended questions.
    - data_file (str): The filename or path of the dataset file to load. This file should be located in the
                       Hugging Face Hub repository specified by the dataset name.

                       For "mcq" type, expected files could be 'mcq-single.csv', 'rodinia-basic.csv', or
                       'rodinia-advanced.csv'.

                       For "open_ended" type, expected files could be 'text.csv' or 'code.csv'.
    - test_mode (bool): If True, only load the first two examples from the dataset for testing purposes.

    Returns:
    - dataset (DatasetDict): A `datasets.DatasetDict` object containing the loaded dataset.

    Raises:
    - ValueError: If the dataset_type is not one of the accepted values.

    Example usage:
    mcq_dataset = load_dataset_from_hub("mcq", "mcq-single.csv")
    open_ended_dataset = load_dataset_from_hub("open_ended", "text.csv")
    """

    # Load MCQ-type dataset from the HPCPerfOpt-MCQA repository on the Hugging Face Hub.
    if dataset_type == "mcq":
        dataset = load_dataset("sharmaarushi17/HPCPerfOpt-MCQA", data_files=data_file)
        #dataset = load_dataset(data_file)

    # Load open-ended dataset from the HPCPerfOpt-Open-ended repository on the Hugging Face Hub.
    elif dataset_type == "open_ended":
        dataset = load_dataset("sharmaarushi17/HPCPerfOpt-Open-ended", data_files=data_file)
        #dataset = load_dataset(data_file)

    # Raise an error if an invalid dataset_type is provided.
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

   # If test_mode is True, only take the first two examples from each split.
    if test_mode:
        for split in dataset.keys():
            dataset[split] = dataset[split].select(range(2))

    return dataset


def load_model_return_response(model_name, prompt):
    """
    Load a question-answering model specified by `model_name` and return the model's response to a given `prompt`.

    This function initializes a pipeline for the 'openmp_question_answering' task using the specified model.
    It then processes a prompt and returns the generated response.

    Parameters:
    - model_name (str): The name of the pre-trained model to be loaded for the pipeline.
    - prompt (str): The input prompt or question to be passed to the model for generating a response.

    Returns:
    - response (dict): A dictionary containing the model's response. The structure of the response dictionary
                       is determined by the underlying model and the task.
    """

    try:
        # Initialize the LM4HPC pipeline with the specified model.
        OMP_QA_model = hpcpipelines(task="openmp_question_answering", model=model_name, pdf_files="", langchain_embedding="")  
        # Generate response using the model.
        response = OMP_QA_model(prompt)

    except Exception as e:
        # Handle exceptions that may occur during model loading or prompt processing.
        print(f"An error occurred: {e}")
        response = None
    
    return response


def create_LLM_prompt_from_example(example,dataset_type, prompt_type, rag):
    '''
    Generate a language model prompt for a given example, based on instructions stored in EVALUATION_PROMPTS dictionary stored in prompts.py

    Parameters:
    - example (dict): A dictionary containing the example data.
    - prompt_type (str): The type of prompt to be generated. Expected values are 'standard', 'cot', 'text', 'code', 'none'.
    - rag(bool): Indicates if the tasks are assisted with retrieval, if true, prompts are formatted accordingly.

    Returns:
    - str: A formatted string that serves as a prompt for a language model, containing the instruction (if provided),
           and the necessary information extracted from the example.

    '''
    # If prompt_type is 'none', set instruction to an empty string
    instruction = ''
    cont = ""
    context = []
    rag_ins = ""

    #RAG prompt formatting
    if rag:
        task_id = example["Source"]
        task_id = str(task_id) + "_doc"
        retrieval = open("../results/retrieval.json", "r")
        corpus = open("../results/corpus.jsonl", "r")
        #Additional information to tell the LM that context is being provided for the task
        rag_ins = " \n Below are additional contexts followed by the task, contexts may or may not help in answering or completing the task. \n"

        try:
            ret = [json.loads(line) for line in retrieval]
            cor = [json.loads(line) for line in corpus]

            #finds the relevant documents for the task and formats it into context
            for line in ret:
                if task_id in line:
                    docs = line[task_id]
                    sorted_docs = sorted(docs.items(), key = lambda x: x[1], reverse = True) #sorts the documents based on their retrieval score
                    top_k = sorted_docs[:3] #leave only the top-k documents retrieved
                    stentries = dict(top_k)
                    keys = list(stentries.keys())

                    lookup = {entry["_id"]: entry for entry in cor}

                    for key in keys:
                        if key in lookup:
                            title = lookup[key].get("title")
                            text = lookup[key].get("text")
                            full = {title: text}
                            context.append(full)
        finally:
            retrieval.close()
            corpus.close()

        for entry in context:
                title = next(iter(entry))
                info = entry[title]
                cont += "\n Context: " + title + "\n" + info + "\n"
        
    if prompt_type != 'none':
        if dataset_type == 'mcq':
            try:
                instruction = EVALUATION_PROMPTS["MCQA"][prompt_type]
            except KeyError:
                raise ValueError(f"Prompt type '{prompt_type}' does not exist for dataset type 'MCQA' in EVALUATION_PROMPTS.")

            question = example['Code Snippet'] + " " + example['Question']
            options = f"A. {example['A']} B. {example['B']} C. {example['C']} D. {example['D']}"
            prompt = f"{instruction} {rag_ins}{cont}{question} Options: {options}"

        elif dataset_type == 'open_ended':
            try:
                instruction = EVALUATION_PROMPTS["OPEN_ENDED"][prompt_type]
            except KeyError:
                raise ValueError(f"Prompt type '{prompt_type}' does not exist for dataset type 'OPEN_ENDED' in EVALUATION_PROMPTS.")
            
            question = example['Code Snippet'] + " " + example['Question']
            prompt = f"{instruction} {rag_ins} {cont} {question}"

        else:
            raise ValueError(f"Invalid dataset type. Expected 'MCQ' or 'OPEN_ENDED', but got '{dataset_type}'.")

    return prompt


def process_cot_response(response_str):
    json_pattern = r"{\'Answer\': \'[A-D]\'}"

    # Find all matches; note that this will only work with this very specific pattern
    matches = re.findall(json_pattern, response_str)

    # If there is at least one match, parse the first one (assuming there's only one JSON object)
    if matches:
        # Convert single quotes to double quotes to make it valid JSON
        json_str = matches[0].replace("\'", "\"")
        response_dict = json.loads(json_str)
        print(f"The extracted answer is: {response_dict['Answer']}")
        return response_dict['Answer']
    else:
        print("No valid JSON object found.")


def bleu_score_evaluation(open_ended_Dataset, model_name, args):
    """
    Evaluate a language model based on bleu score between the model's response and the correct answer for open response questions.

    Parameters:
    - open_ended_Dataset (dict): A dataset containing open ended questions and answers.
    - model_name (str): The name of the model to be evaluated.
    - args (Namespace): An argparse Namespace containing 'dataset_type' and 'prompt_type'.

    Returns:
    - accuracy (float): The proportion of correctly answered questions.
    - results (list): A list of dictionaries containing prompts, responses, correct answers, and correctness flags.
    """
    sum = 0
    total = 0
    results = []
    weights = (1.0, 0.0, 0.0) #ngram weights, position indicates n, value is weights
    for idx, example in enumerate(open_ended_Dataset['train']):
        print(f"Example #{idx + 1}: {example}\n{'-' * 80}")
        correct_answer = example['Answer']
        prompt = create_LLM_prompt_from_example(example, args.dataset_type, args.prompt_type, args.rag)
        if prompt is not None:
            print(f"Prompt #{idx + 1}:\n{prompt}\n{'-' * 80}")
            response = load_model_return_response(model_name, prompt)
            if response:
                response = response.replace("\n", "")
            if response:  # Check if response is not empty
                print("response",response)
                response_type = type(response)
                print(f"The type of 'response' is: {response_type}")
                if correct_answer is not None:
                    reference = [correct_answer.split()]
                    candidate = response.split()
                    score = sentence_bleu(reference, candidate, weights=weights)
                    sum += score

                    is_correct = False
                    if (score > 0.35):
                        is_correct = True

                    result_dict = {
                        "prompt": prompt,
                        "response": response,
                        "correct_answer": correct_answer,
                        "score": score,
                        "is_correct": is_correct
                    }
                    results.append(result_dict)
                else:
                    print("Error: 'Answer' key is missing in the example.")
            else:
                print("Response is empty or None.")
        else:
            print("Prompt could not be generated due to missing data.")
        total += 1  # Increment total inside the loop
    accuracy = sum / total if total > 0 else 0  # Prevent division by zero
    return accuracy, results


def codebertscore_evaluation(dataset, model_name, args):
    """TODO"""

def exact_match_evaluation(mcqa_dataset, model_name, args):
    """
    Evaluate a language model based on exact match between the model's response and the correct answer for multiple-choice questions.

    Parameters:
    - mcqa_dataset (dict): A dataset containing multiple-choice questions and answers.
    - model_name (str): The name of the model to be evaluated.
    - args (Namespace): An argparse Namespace containing 'dataset_type' and 'prompt_type'.

    Returns:
    - accuracy (float): The proportion of correctly answered questions.
    - results (list): A list of dictionaries containing prompts, responses, correct answers, and correctness flags.
    """

    correct = 0
    total = 0
    results = []
    for idx, example in enumerate(mcqa_dataset['train']):
        print(f"Example #{idx + 1}: {example}\n{'-' * 80}")
        correct_answer = example['Answer']
        prompt = create_LLM_prompt_from_example(example, args.dataset_type, args.prompt_type, args.rag)
        if prompt is not None:
            print(f"Prompt #{idx + 1}:\n{prompt}\n{'-' * 80}")
            response = load_model_return_response(model_name, prompt)
            if response:
                response = response.replace("\n", "")
            if response:  # Check if response is not empty
                print("response",response)
                response_type = type(response)
                print(f"The type of 'response' is: {response_type}")
                if correct_answer is not None:
                    if args.prompt_type=="standard":
                        is_correct = (correct_answer in response)
                    elif args.prompt_type=="cot":
                        #get answer from json object and compare
                        ans = process_cot_response(response)
                        if ans is None:
                            is_correct=False
                        else:
                            is_correct = (correct_answer.strip() == ans.strip())  # Strip spaces before comparison
                    else:
                        print("error: exact match prompt_type invalid")
                    if is_correct:
                        correct += 1
                    # Now we collect the results after each example is processed

                    result_dict = {
                        "prompt": prompt,
                        "response": response,
                        "correct_answer": correct_answer,
                        "is_correct": is_correct
                    }
                    results.append(result_dict)
                else:
                    print("Error: 'Answer' key is missing in the example.")
            else:
                print("Response is empty or None.")
        else:
            print("Prompt could not be generated due to missing data.")
        total += 1  # Increment total inside the loop
    accuracy = correct / total if total > 0 else 0  # Prevent division by zero
    return accuracy, results


def semantic_similarity_evaluation(open_ended_dataset, model_name, args):
    """
    Evaluate a language model based on semantic similarity between the model's response and the correct answer for open-ended questions.

    This function computes the semantic similarity using cosine similarity between the embeddings of the model's response and the correct answer. A threshold is set to determine whether the response is considered 'correct'.

    Parameters:
    - open_ended_dataset (dict): A dataset containing open-ended questions and answers.
    - model_name (str): The name of the model to be evaluated.
- args (Namespace): An argparse Namespace containing 'dataset_type' and 'prompt_type'.

    Returns:
    - accuracy (float): The proportion of questions where the cosine similarity between the response and the correct answer is above the threshold.
    - results (list): A list of dictionaries containing prompts, responses, correct answers, cosine similarity values, and correctness flags.

    """
    # Initialize sentence transformer model
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    correct_count = 0

    results = []
    for idx, example in enumerate(open_ended_dataset['train']):
        print(f"Example #{idx + 1}: {example}\n{'-' * 80}")
        #correct_answer = example['Answer']
        correct_answer = example["code"]

        prompt = create_LLM_prompt_from_example(example, args.dataset_type, args.prompt_type, args.rag)

        if prompt is not None:
            print(f"Prompt #{idx + 1}:\n{prompt}\n{'-' * 80}")
            response = load_model_return_response(model_name, prompt)
            if response:  # Check if response is not empty
                response = response.replace("\n", "")
                print("response",response)
                response_type = type(response)
                print(f"The type of 'response' is: {response_type}")

                # Generate embeddings for the response and correct answer
                response_embedding = embedder.encode(response, convert_to_tensor=True)
                correct_answer_embedding = embedder.encode(correct_answer, convert_to_tensor=True)

                # Compute cosine similarity
                cosine_similarity = util.pytorch_cos_sim(response_embedding, correct_answer_embedding).item()

                is_correct = cosine_similarity >= 0.3  # Adjust the threshold as needed

                if is_correct:
                    correct_count += 1

            results.append({
                "prompt": prompt,
                "response": response,
                "correct_answer": correct_answer,
                "similarity": cosine_similarity,
                "is_correct": is_correct
            })

            #Add model_name to store in csv

    num_rows = len(open_ended_dataset['train'])
    accuracy = correct_count / num_rows if num_rows > 0 else 0
    return accuracy, results


def llm_as_a_judge_evaluation(open_dataset, model_name, args):
    """
    Evaluate a language model based on exact match between the model's response and the correct answer for multiple-choice questions.

    Parameters:
    - mcqa_dataset (dict): A dataset containing multiple-choice questions and answers.
    - model_name (str): The name of the model to be evaluated.
    - args (Namespace): An argparse Namespace containing 'dataset_type' and 'prompt_type'.

    Returns:
    - accuracy (float): The proportion of correctly answered questions.
    - results (list): A list of dictionaries containing prompts, responses, correct answers, and correctness flags.
    """

    correct = 0
    total = 0
    results = []
    for idx, example in enumerate(open_dataset['train']):
        print(f"Example #{idx + 1}: {example}\n{'-' * 80}")
        correct_answer = example['Answer']
        prompt = create_LLM_prompt_from_example(example, args.dataset_type, args.prompt_type)
        if prompt is not None:
            print(f"Prompt #{idx + 1}:\n{prompt}\n{'-' * 80}")
            response = load_model_return_response(model_name, prompt)
            if response:  # Check if response is not empty
                print("response",response)
                response_type = type(response)
                print(f"The type of 'response' is: {response_type}")
                if correct_answer is not None:
                    if args.prompt_type=="standard":
                        #create prompt to use gpt-4 as a judge and evaluate the response
                        judge_prompt = "Given the following ground truth answer and llm-generted response, determine whether the llm-generated response is correct in relation to the provided ground truth. Only output Y for Yes and N for No. Correct answer: " + correct_answer + "Response: " +  response 
                        is_correct = load_model_return_response("gpt-4", judge_prompt)
                    elif args.prompt_type=="cot":
                        #get answer from json object and compare
                        ans = process_cot_response(response)
                        if ans is None:
                            is_correct=False
                        else:
                            is_correct = (correct_answer.strip() == ans.strip())  # Strip spaces before comparison
                    else:
                        print("error: llm-as-a-judge prompt_type invalid")
                    if is_correct:
                        correct += 1
                    # Now we collect the results after each example is processed
                    result_dict = {
                        "prompt": prompt,
                        "response": response,
                        "correct_answer": correct_answer,
                        "is_correct": is_correct
                    }
                    results.append(result_dict)
                else:
                    print("Error: 'Answer' key is missing in the example.")
            else:
                print("Response is empty or None.")
        else:
            print("Prompt could not be generated due to missing data.")
        total += 1  # Increment total inside the loop
    accuracy = correct / total if total > 0 else 0  # Prevent division by zero
    return accuracy, results


def store_eval_results_in_csv(dataset_type, data_file, prompt_type, eval_type, model_name, results, overall_accuracy, rag, overwrite=True):
    """
    Store evaluation results in a CSV file.

    This function creates a CSV file and directory structure based on the given parameters. It handles different types
    of evaluation results and ensures that the resulting CSV file's headers match the results data structure. If the
    CSV file already exists, it will not be overwritten unless `overwrite` is set to True.

    Parameters:
    - dataset_type (str): The type of dataset being evaluated (e.g., "mcq", "open-ended").
    - data_file (str): The name of the data file (e.g., "mcq-single.csv").
    - prompt_type (str): The type of prompt used in the evaluation (e.g., "standard").
    - eval_type (str): The type of evaluation ("exact_match" or "semantic_similarity").
    - model_name (str): The name of the model being evaluated (e.g., "gpt-4").
    - results (list of dicts): A list of dictionaries containing the results data.
    - overwrite (bool, optional): Whether to overwrite the existing CSV file if it exists. Defaults to False.

    Returns:
    None: This function does not return any value. It writes directly to a file.

    Raises:
    Exception: If an error occurs during the creation of the directories or the writing of the CSV file.
    """

    try:
        # Define the directory structure for Results->{dataset_type}->{data_file}->{model_name}
        directory_path = os.path.join("Results", dataset_type, data_file, model_name)
        os.makedirs(directory_path, exist_ok=True)  # Create the directory if it doesn't exist

        # Define the CSV file path
        mname = model_name.replace("/", "_")
        if rag:
            output_csv = os.path.join(directory_path, f"{dataset_type}_{mname}_{prompt_type}_rag_{eval_type}.csv")
        else:
            output_csv = os.path.join(directory_path, f"{dataset_type}_{mname}_{prompt_type}_{eval_type}.csv")

        # Check if file exists and if overwrite is False
        if os.path.exists(output_csv) and not overwrite:
            print(f"The file {output_csv} already exists. Set overwrite=True to overwrite it.")
            return

        # Determine the headers based on the eval_type
        headers = ['Prompt', 'Response', 'Correct Answer']
        if eval_type == "exact_match":
            headers += ['Is Correct']
        elif eval_type == "semantic_similarity":
            headers += ['Similarity', 'Is Correct']
        elif eval_type == "bleu_score":
            headers += ['Score', 'Is Correct']

        # Write the results to the CSV file
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)  # Write the headers
            
            # Write the data
            for result in results:
                if eval_type == "exact_match":
                    writer.writerow([result['prompt'], result['response'], result['correct_answer'], result['is_correct']])
                elif eval_type == "semantic_similarity":
                    writer.writerow([result['prompt'], result['response'], result['correct_answer'], result['similarity'], result['is_correct']])
                elif eval_type == "bleu_score":
                    writer.writerow([result['prompt'], result['response'], result['correct_answer'], result['score'], result['is_correct']])
            
            # Append overall accuracy
            writer.writerow([])  # Add an empty row for separation
            writer.writerow(['Overall Accuracy', overall_accuracy])

    except Exception as e:
        print(f"An error occurred: {e}")


def get_finetuned_model():
    '''Load finetuned codellama model '''


def main(args):
    # Your code to load the dataset and evaluate the model would go here
    print(f"Loading {args.dataset_type} dataset from file: {args.data_file}")
    print(f"Using prompt type: {args.prompt_type}")
    print(f"Evaluating model(s): {', '.join(args.model_names)}")
    
    dataset = load_dataset_from_hub(args.dataset_type, args.data_file, args.test_mode)
    print(f"Loaded dataset with {len(dataset['train'])} examples for evaluation.",dataset) 
    
    #need a loop for model_names
    for model_name in args. model_names:
        if args.eval_type=="exact_match":
            accuracy, results = exact_match_evaluation(dataset, model_name, args)
            store_eval_results_in_csv(args.dataset_type, args.data_file, args.prompt_type, args.eval_type, model_name, results, accuracy, args.rag)
            print(f"Model: {model_name} - Accuracy: {accuracy}")
        elif args.eval_type=="semantic_similarity":
            accuracy, results = semantic_similarity_evaluation(dataset, model_name, args)
            store_eval_results_in_csv(args.dataset_type, args.data_file, args.prompt_type, args.eval_type, model_name, results, accuracy, args.rag)
            print(f"Model: {model_name} - Accuracy: {accuracy}")
        elif args.eval_type=="bleu_score":
            accuracy, results = bleu_score_evaluation(dataset, model_name, args)
            store_eval_results_in_csv(args.dataset_type, args.data_file, args.prompt_type, args.eval_type, model_name, results, accuracy, args.rag)
            print(f"Model: {model_name} - Accuracy: {accuracy}")
        elif args.eval_type=="codebertscore":
            accuracy, results = codebertscore_evaluation(dataset, model_name, args)
            store_eval_results_in_csv(args.dataset_type, args.data_file, args.prompt_type, args.eval_type, model_name, results, accuracy, args.rag)
            print(f"Model: {model_name} - Accuracy: {accuracy}")
        elif args.eval_type=="llm-as-judge":
            accuracy, results = llm_as_a_judge_evaluation(dataset, model_name, args)
            store_eval_results_in_csv(args.dataset_type, args.data_file, args.prompt_type, args.eval_type, model_name, results, accuracy, args.rag)
            print(f"Model: {model_name} - Accuracy: {accuracy}")
        else:
            print("error")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Evaluate a model using a dataset from the Hugging Face Hub.')

    # Add the arguments
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode to load a smaller part of the dataset.')

    parser.add_argument('--rag', action='store_true', help='Run evaluation with retrieval augmentation')

    parser.add_argument('--dataset_type',
                        type=str,
                        required=True,
                        choices=['mcq', 'open_ended'],
                        help='The type of dataset to load. '
                             'Use "mcq" for multiple choice questions or "open_ended" for open-ended questions.')

    parser.add_argument('--data_file',
                        type=str,
                        required=True,
                        help='The filename of the dataset file to load from the Hugging Face Hub repository. '
                             'For "mcq" type, possible values include "mcq-single.csv", "rodinia-basic.csv", or "rodinia-advanced.csv". '
                             'For "open_ended" type, possible values include "text.csv" or "code.csv".')

    parser.add_argument('--model_names',
                        type=str,
                        nargs='+',
                        required=True,
                        help='The name(s) of the model(s) to use for evaluation. Multiple model names can be provided separated by spaces.')

    parser.add_argument('--prompt_type',
                        type=str,
                        default='none',  # Set the default value to 'none'
                        choices=['standard', 'cot', 'text', 'code', 'rag', 'none'],
                        help='The type of prompt to be used for the LLM. "none" will use no additional prompt information.')
    
    parser.add_argument('--eval_type',
                        type=str,
                        default='none',  # Set the default value to 'none'
                        choices=['exact_match','semantic_similarity','bleu_score','codebertscore','llm_as_a_judge'],
                        help='The type of evaluation to be performed. "none" will use no additional prompt information.')

    args = parser.parse_args()
    main(args)
