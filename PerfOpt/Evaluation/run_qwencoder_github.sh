#!/bin/bash

models=(
#  "Qwen/Qwen2.5-Coder-0.5B-Instruct"
#  "Qwen/Qwen2.5-Coder-1.5B-Instruct"
#  "Qwen/Qwen2.5-Coder-3B-Instruct"
#  "Qwen/Qwen2.5-Coder-7B-Instruct"
)
retrieval_paths=(
  "datasets/github/GIST-Embedding-v0.json"
#  "datasets/github/st-codesearch-distilroberta-base.json"
)
corpus_path="datasets/github/corpus.jsonl"

for model in "${models[@]}"; do
    for retrieval in "${retrieval_paths[@]}"; do
        echo "Starting evaluation for model: $model with retrieval path: $retrieval"

        python evals.py \
            --dataset_type code_generation \
            --data_file queries.jsonl \
            --model_names "$model" \
            --prompt_type standard \
            --eval_type static_code_metric \
            --rag \
            --k_documents 3 \
            --retrieval_path "$retrieval" \
            --corpus_path "$corpus_path" \
            --load_in_4bit \
            --dataset_name datasets/polybench

        echo "Finished evaluation for model: $model with retrieval path: $retrieval"
        echo "----------------------------------------"
    done
done