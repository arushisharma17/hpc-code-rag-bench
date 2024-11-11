## Directory Structure

Results
└── [Dataset Type]
    └── [Data File]
        └── [Model Name]
            └── CSV Files
### Description
- **Results**: The root directory where all evaluation results are stored.
- **[Dataset Type]**: A subdirectory named after the type of dataset being evaluated (e.g., "mcq", "open-ended"). This allows for separate storage of results based on the dataset.
- **[Data File]**: Under each dataset type, there is a subdirectory for each data file used in the evaluations (e.g., "mcq-single.csv"). This level of organization helps in distinguishing results obtained from different data sources or configurations.
- **[Model Name]**: Inside each data file directory, there are further subdirectories named after the models evaluated (e.g., "gpt-4"). This categorization facilitates easy access to results specific to each model.
- **CSV Files**: Within each model's directory, the results are stored in CSV files. The file names are constructed based on the dataset type, model name, prompt type, and evaluation type, following the format: `[Dataset Type]_[Model Name]_[Prompt Type]_[Eval Type].csv`.

### File Naming and Access
- **File Naming**: The CSV files are named to reflect the specific evaluation context, making it straightforward to identify the contents.
- **Overwrite Protection**: By default, if a CSV file with the given name already exists, it will not be overwritten. You can set the `overwrite` parameter to `True` in the function call to overwrite existing files if needed.


### Results Table

| Dataset Type | Data File | Model Name | Semantic Similarity | Exact Match | BLEU Score |
|--------------|-----------|------------|---------------------|-------------|------------|
| MCQ | mcq-single.csv | gpt-4 | | | |
| MCQ | mcq-single.csv | gpt-3.5-turbo | | 0.6705882352941176| |
| MCQ | mcq-single.csv | codellama/CodeLlama-7b-hf | | | |
| MCQ | mcq-single.csv | HuggingFaceH4/starchat-beta | | | |
| MCQ | mcq-rodinia-basic.csv | gpt-4 | | | |
| MCQ | mcq-rodinia-basic.csv | gpt-3.5-turbo | | 0.95 | |
| MCQ | mcq-rodinia-basic.csv | codellama/CodeLlama-7b-hf | | | |
| MCQ | mcq-rodinia-basic.csv | HuggingFaceH4/starchat-beta | | | |
| MCQ | mcq-rodinia-advanced.csv | gpt-4 | |0.95 | |
| MCQ | mcq-rodinia-advanced.csv | gpt-3.5-turbo | | 0.85| |
| MCQ | mcq-rodinia-advanced.csv | codellama/CodeLlama-7b-hf | | | |
| MCQ | mcq-rodinia-advanced.csv | HuggingFaceH4/starchat-beta | | | |
| Open-ended | text.csv | gpt-4 |0.896551724137931 | | |
| Open-ended | text.csv | gpt-3.5-turbo | 0.896551724137931 | | |
| Open-ended | text.csv | codellama/CodeLlama-7b-hf | | | |
| Open-ended | text.csv | HuggingFaceH4/starchat-beta | | | |
| Open-ended | code.csv | gpt-4 | 0.9206349206349206 | | |
| Open-ended | code.csv | gpt-3.5-turbo | | | |
| Open-ended | code.csv | codellama/CodeLlama-7b-hf | Accuracy: 0.9841269841269841| | |
| Open-ended | code.csv | HuggingFaceH4/starchat-beta | 0.9682539682539683 | | | 
| Open-ended | rodinia-open-ended-advanced.csv | gpt-4 | 1.0 | | |
| Open-ended | rodinia-open-ended-advanced.csv | gpt-3.5-turbo | | | |
| Open-ended | rodinia-open-ended-advanced.csv | codellama/CodeLlama-7b-hf | 0.3| | |
| Open-ended | rodinia-open-ended-advanced.csv | HuggingFaceH4/starchat-beta | 0.0 | | |
| Open-ended | rodinia-open-ended-basic.csv | gpt-4 | 1.0| | |
| Open-ended | rodinia-open-ended-basic.csv | gpt-3.5-turbo | 1.0 | | |
| Open-ended | rodinia-open-ended-basic.csv | codellama/CodeLlama-7b-hf | 0.62| | |
| Open-ended | rodinia-open-ended-basic.csv | HuggingFaceH4/starchat-beta | 0.0 | | |




