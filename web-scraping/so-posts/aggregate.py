import os
import json


def read_json_files(directory):
    """Reads all JSON files from the specified directory and merges them into one list."""
    merged_data = []

    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return merged_data

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        merged_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error reading {filename}: {e}")

    return merged_data


if __name__ == "__main__":
    directory = "."
    merged_posts = read_json_files(directory)
    with open(f"so-posts-all.json", "w") as f:
        json.dump(merged_posts, f, indent=2)
    # print(f"Merged {len(merged_posts)} JSON entries.")
