import code_bert_score
import json
import torch


# load predictions
cands = []
with open('llm-code-output/llama-3.2-3B/without-context/codegen-meta-llamaLlama-3.2-3B-Instruct-without-rag-1741717042433.json', "r") as file:
    predictions = json.load(file)
    for row in predictions:
        cands.append(row['code'])

# load ground truth
refs = []
with open('llm-code-output/polybench-ground-truth.json', "r") as file:
    corpus = json.load(file)
    for row in corpus:
        refs.append(row['correct_answer'])

pred_results = code_bert_score.score(cands=cands, refs=refs, lang='c')

# print('raw pred_results', pred_results)

tensors = [pred_results[0], pred_results[1], pred_results[2], pred_results[3]]
means = [tensor.mean() for tensor in tensors]
print(f"precision: {means[0].item()}, recall: {means[1].item()}, f1: {means[2].item()}, f3: {means[3].item()}")