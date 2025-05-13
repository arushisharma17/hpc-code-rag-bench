import json
import jsonlines


def to_corpus(input_file, output_file):
    data = []
    unanswered = 0
    with open(input_file, "r") as file:
        posts = json.load(file)
        print(f"Loaded {len(posts)} stackoverflow posts...")

        for question in posts:
            entry = {
                '_id': question['url'],
                'title': question['question'],
                'text': question['body'] + '\n',
                'metadata': {
                    'url': question['url']
                }
            }
            answers = ''
            for answer in question['answers']:
                answers += f"A: {answer['body']}"
                answers += '\n'
            entry['text'] += answers
            data.append(entry)

            if len(question['answers']) == 0:
                unanswered += 1

    with jsonlines.open(output_file, mode = "w") as writer:
        writer.write_all(data)

    with open(f"so-posts-corpus.json", "w") as f:
        json.dump(data, f, indent=2)

    print("unanswered posts", unanswered)


if __name__ == "__main__":
    print("STARTING")
    to_corpus("../../web-scraping/so-posts/so-posts-all.json", "so-posts-corpus.jsonl")
    print("DONE")