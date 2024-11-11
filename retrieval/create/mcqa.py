import os
import argparse
import datasets
from tqdm import tqdm
import jsonlines
import csv
import os

def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode = "w") as writer:
        writer.write_all(data)

def save_tsv_dict(data, fp, fields):
    dir_path = os.path.dirname(fp)
    os.makedirs(dir_path, exist_ok = True)

    with open(fp, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fields, delimiter = "\t", lineterminator = "\n")
        writer.writeheader()
        writer.writerows(data)

def d2d(data, split):
    data = data[split]
    queries, docs, qrels = [], [], []

    for item in tqdm(data):
        doc = item["Code Snippet"] + "\n" + item["Question"] + "\n" "Option A: " + item["A"] + "\n" "Option B: " + item["B"] + "\n" "Option C: " + item["C"] + "\n" "Option D: " + item["D"]
        code = item["Code Snippet"] + "\n" + item["Question"] + "\n" "Option A: " + item["A"] + "\n" "Option B: " + item["B"] + "\n" "Option C: " + item["C"] + "\n" "Option D: " + item["D"] + "\n" + "Correct Answer: " + item["Answer"]
        doc_id = "{Source}_doc".format_map(item)
        code_id = "{Source}_code".format_map(item)

        queries.append({"_id": doc_id, "text": doc, "metadata": {}})
        docs.append({"_id": code_id, "title": item["Question"], "text": code, "metadata": {}})
        qrels.append({"query-id": doc_id, "corpus-id": code_id, "score": 1})

    return queries, docs, qrels
        

def main():
    dataset = datasets.load_dataset(args.dataset, data_files = "mcq-single.csv")
    path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(path, exist_ok = True)
        
    docs, queries = [], []

    for split in args.splits:
        queries_split, docs_split, qrels_split = d2d(dataset, split)
        docs += docs_split
        queries += queries_split

        qrels_path = os.path.join(path, "qrels", f"test.tsv")
        save_tsv_dict(qrels_split, qrels_path, ["query-id", "corpus-id", "score"])

    if (args.cor == "both"):
        save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
        save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))
    elif(args.cor == "corpus"):
        save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))
    else:
        save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default = "sharmaarushi17/HPCPerfOpt-MCQA")
    parser.add_argument("--splits", type = str, default = ["train"])
    parser.add_argument("--output_name", type = str, default = "datastore")
    parser.add_argument("--output_dir", type = str, default = "datasets")
    parser.add_argument("--cor", type = str, default = "both", choices = ["corpus", "query", "both"])

    args = parser.parse_args()
    main()