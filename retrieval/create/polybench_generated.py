import csv
import json
import jsonlines
import os


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


def to_query(input_file):
    queries = []
    corpus = []
    qrels = []
    with open(input_file, "r") as file:
        data = json.load(file)
        print(f"Loaded {len(data)} polybench queries...")

        for query in data:
            query_id = f"PolyBench/Query/{query['id']}"
            corpus_id = f"PolyBench/Corpus/{query['id']}"
            queries.append({
                '_id': query_id,
                'text': query['kernel_description'],
                'metadata': {
                    'name': query['name'],
                    'path': query['path']
                }
            })
            corpus.append({
                '_id': corpus_id,
                'title': query['name'],
                'text': query['kernel_description'],
                'correct_answer': query['kernel'],
                'metadata': {
                    'name': query['name'],
                    'path': query['path']
                }
            })
            qrels.append({'query-id': query_id, 'corpus-id': corpus_id, 'score': 1})

    path = os.path.join('datasets', 'datastore')
    save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
    save_file_jsonl(corpus, os.path.join(path, "corpus.jsonl"))
    qrels_path = os.path.join(path, "qrels", f"test.tsv")
    save_tsv_dict(qrels, qrels_path, ["query-id", "corpus-id", "score"])


if __name__ == "__main__":
    print("STARTING")
    to_query('polybench/polybench_generated.json')
    print('DONE')
