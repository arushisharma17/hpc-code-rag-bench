#!/bin/bash

models=(
  "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
)
retrieval_paths=(
  "datasets/simple-openmp/github-latest/GIST-Embedding-v0.json"
  "datasets/simple-openmp/github-latest/st-codesearch-distilroberta-base.json"
)
corpus_path="datasets/simple-openmp/github-latest/corpus.jsonl"

for model in "${models[@]}"; do
    for retrieval in "${retrieval_paths[@]}"; do
        echo "Starting evaluation for model: $model with retrieval path: $retrieval"

        python evals.py \
            --dataset_type code_generation \
            --data_file queries.jsonl \
            --model_names "$model" \
            --prompt_type simple-openmp \
            --eval_type unit_test_execution \
            --rag \
            --k_documents 3 \
            --retrieval_path "$retrieval" \
            --corpus_path "$corpus_path" \
            --load_in_4bit \
            --dataset_name datasets/simple-openmp \
            --test_mode

        echo "Finished evaluation for model: $model with retrieval path: $retrieval"
        echo "----------------------------------------"
    done
done