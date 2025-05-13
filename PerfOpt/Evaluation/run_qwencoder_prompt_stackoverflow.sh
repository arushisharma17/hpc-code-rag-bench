#!/bin/bash

models=(
  "Qwen/Qwen2.5-Coder-0.5B-Instruct"
  "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  "Qwen/Qwen2.5-Coder-3B-Instruct"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
)
retrieval_paths=(
  "datasets/stackoverflow/GIST-Embedding-v0.json"
  "datasets/stackoverflow/st-codesearch-distilroberta-base.json"
)
corpus_path="datasets/stackoverflow/corpus.jsonl"

for model in "${models[@]}"; do
    for retrieval in "${retrieval_paths[@]}"; do
        echo "Starting evaluation for model: $model with retrieval path: $retrieval"

        python evals.py \
            --dataset_type code_generation \
            --data_file queries.jsonl \
            --model_names "$model" \
            --prompt_type simple-openmp \
            --eval_type static_code_metric \
            --rag \
            --k_documents 1 \
            --retrieval_path "$retrieval" \
            --corpus_path "$corpus_path" \
            --load_in_4bit \
            --dataset_name datasets/polybench

        echo "Finished evaluation for model: $model with retrieval path: $retrieval"
        echo "----------------------------------------"
    done
done