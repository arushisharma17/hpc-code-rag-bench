from codebleu import calc_codebleu
import json

if __name__ == "__main__":
    predictions = []
    with open(
            'llm-code-output/llama-3.2-3B/without-context/codegen-meta-llamaLlama-3.2-3B-Instruct-without-rag-1741717042433.json',
            "r") as file:
        codes = json.load(file)
        for row in codes:
            predictions.append(row['code'])

    refs = []
    with open('llm-code-output/polybench-ground-truth.json', "r") as file:
        corpus = json.load(file)
        for row in corpus:
            refs.append(row['correct_answer'])

    result = calc_codebleu(references=refs, predictions=predictions, lang="c", weights=(0.10, 0.10, 0.40, 0.40), tokenizer=None)
    print(result)