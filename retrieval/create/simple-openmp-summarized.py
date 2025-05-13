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
        print(f"Loaded {len(data)} simple openmp queries...")
        id = 1
        for query in data:
            if query['summarized_code'] is None:
                continue
            query_id = f"SimpleOpenMpSummarized/Query/{id}"
            corpus_id = f"SimpleOpenMpSummarized/Corpus/{id}"
            queries.append({
                '_id': query_id,
                'text': query['summarized_code'],
                'metadata': {
                    "original_id": query['original_id']
                }
            })
            corpus.append({
                '_id': corpus_id,
                'text': query['summarized_code'],
                'metadata': {
                    "original_id": query['original_id']
                }
            })
            qrels.append({'query-id': query_id, 'corpus-id': corpus_id, 'score': 1})
            id += 1

    path = os.path.join('datasets', 'datastore')
    save_file_jsonl(queries, os.path.join(path, "simple-openmp-summarized-queries.jsonl"))
    save_file_jsonl(corpus, os.path.join(path, "simple-openmp-summarized-corpus.jsonl"))
    qrels_path = os.path.join(path, "qrels", f"test.tsv")
    save_tsv_dict(qrels, qrels_path, ["query-id", "corpus-id", "score"])
    with open(f"simple-openmp-summarized-corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)


if __name__ == "__main__":
    print("STARTING")
    to_query('code-summarize/simple-code.json')
    print('DONE')
