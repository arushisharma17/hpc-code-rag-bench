import csv
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


def csv_to_json(csv_file):
    queries = []
    corpus = []
    qrels = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        idx = 0
        for row in reader:
            query_id = f"Github-OpenMP/Query/{idx}"
            corpus_id = f"Github-OpenMP/Corpus/{idx}"
            queries.append({
                '_id': query_id,
                'text': row["code"],
                'metadata': {}
            })
            corpus.append({
                '_id': corpus_id,
                'title': '',
                'text': row["code"],
                'metadata': {}
            })
            qrels.append({'query-id': query_id, 'corpus-id': corpus_id, 'score': 1})
            idx += 1

    path = os.path.join('datasets', 'datastore')
    save_file_jsonl(queries, os.path.join(path, "queries-github-openmp-new.jsonl"))
    save_file_jsonl(corpus, os.path.join(path, "corpus-github-openmp-new.jsonl"))
    qrels_path = os.path.join(path, "qrels", f"test-github-openmp-new.tsv")
    save_tsv_dict(qrels, qrels_path, ["query-id", "corpus-id", "score"])


if __name__ == "__main__":
    csv_file = "github/GH_openmp_new.csv"
    csv_to_json(csv_file)
