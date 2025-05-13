## Diagrams

#### Scraping the documents from Stackoverflow and Github

![scrape-stackoverflow-1](figures/scrape-stackoverflow-1.png "scrape-stackoverflow-1")
![scrape-stackoverflow-2](figures/scrape-stackoverflow-2.png "scrape-stackoverflow-2")

Raw Stackoverflow posts are stored in `web-scraping/so-posts` folder.

Stackoveflow corpus file for BEIR is in `retrieval/create/stackoverflow` folder.
Github corpus file for BEIR is in `retrieval/create/github` folder.

#### Generating embeddings and similarity scores using BEIR

![embedding-pipeline](figures/embedding-pipeline.png "embedding-pipeline")


#### Inference

![inference-pipeline](figures/inference-pipeline.png "inference-pipeline")


## Running the program

### Setup
We maintain two Python environments. One for generating a retrieval file (mapping of scores) and one for code generation (inference).

Environment for generating retrieval file is named `coms599-3.10`:

```shell
conda create -n coms599-3.10 python=3.10
pip install -r requirements.txt
```

Environment for inference is named `lm4hpc`:

```shell
conda create -n lm4hpc python=3.10
cd PerfOpt
pip install -r requirements.txt
```

### Generating retrieval file

Activate `coms599-3.10` environment:

```shell
conda activate coms599-3.10
```

Add your dataset name in `retrieval/run_sbert_retriever.sh` file. For example, we want to run retrieval with Polybench query and Stackoverflow corpus so we use `polybench-w-stackoverflow-corpus`:
```shell
# retrieval/run_sbert_retriever.sh
# replace here
for dataset_name in "polybench-w-stackoverflow-corpus"
do
  ...
  ...
```

Then run it by passing model name, model tag, and batch size as arguments:
```shell
./run_sbert_retriever.sh avsolatorio/GIST-Embedding-v0 GIST-Embedding-v0 64
```

The retrieval file is saved in `retrieval/results` folder. Keep note of this file as we will use this in inference.

### Inference

Activate `lm4hpc` environment:

```shell
conda activate lm4hpc
```

The Polybench and SimpleOpenMP queries are in `PerfOpt/Evaluation/datasets/polybench` and `PerfOpt/Evaluation/datasets/simple-openmp`, respectively. Copy the retrieval file to the respective folder. For example, in `datasets/polybench` we have `GIST-Embedding-v0.json` retrieval file.

Running inference:

There are some prepared files in `PerfOpt/Evaluation/run_qwencoder_*.sh`. Here we run Polybench with Stackoverflow corpus using Qwen Coder LLM. We use static code metrics codebleu and codebert score.
```shell
cd PerfOpt/Evaluation
./run_qwencoder_stackoverflow.sh
```

The output is saved in `PerfOpt/Evaluation/codegen-output` folder.

An example where we use unit testing (program execution) can be seen in file `PerfOpt/Evaluation/run_codegen_simple_*.sh`. It uses the SimpleOpenMP dataset rather than Polybench.

You may run other LLMs, queries and corpus by building on top of that file. We have a complete scripts for 11 LLMs in another section below.

Running unit test execution for SimpleOpenMP with both Github and Stackoverflow retrieval files:

```shell
python evals.py \
    --dataset_type code_generation \
    --data_file queries.jsonl \
    --model_names "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
    --prompt_type simple-openmp \
    --eval_type unit_test_execution \
    --rag \
    --k_documents 3 \
    --retrieval_path "datasets/simple-openmp/github-latest/GIST-Embedding-v0.json" \
    --retrieval_path_second "datasets/simple-openmp/stackoverflow/GIST-Embedding-v0.json" \
    --corpus_path "datasets/simple-openmp/github-latest/corpus.jsonl" \
    --corpus_path_second "datasets/simple-openmp/stackoverflow/corpus.jsonl" \
    --load_in_4bit \
    --dataset_name datasets/simple-openmp
```

### Sbatch Scripts

Go to `sbatch-scripts` folder:
```shell
cd `sbatch-scripts`
```

Python env installation:
```shell
sbatch --export=PROJECT_DIR="/work/classtmp/azhar/projects/hpc-code-rag-bench",MICROMAMBA_ROOT_PATH="/work/classtmp/azhar/micromamba",MICROMAMBA_ENV_NAME="hpccoderag",HF_HOME_PATH="/work/classtmp/azhar/hf" \
  --partition="instruction" \
  --account="s2025.coms.599.3" \
  --mail-user="azhar@iastate.edu" setup.sh
```

Running inference which will schedule all shell files in `sbatch-scripts/codegen_*.sh`:
```shell
./run_codegen.sh --partition="instruction" --account="s2025.coms.599.3" --gres="gpu:a100:1" --mail-user="azhar@iastate.edu" --project-dir="/work/classtmp/azhar/projects/hpc-code-rag-bench" --micromamba-root-path="/work/classtmp/azhar/micromamba" --micromamba-env-name="hpccoderag" --hf-home-path="/work/classtmp/azhar/hf"
```