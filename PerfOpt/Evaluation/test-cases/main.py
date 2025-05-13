import json
from pathlib import Path
import re
import subprocess

if __name__ == "__main__":
    search_dir = Path('./tests')
    cpp_files = list(search_dir.rglob('*.cpp'))

    generated_code = []
    code = {}

    success = 0
    fail = 0

    with open('../codegen-output/QwenQwen2.5-Coder-1.5B-Instruct-1747088958074.json', 'r') as f:
        generated_code = json.load(f)
        code = {
            item["_id"].split("/")[-1]: item
            for item in generated_code['code_gens']
        }

    for file_path in cpp_files:
        print(f"Opening: {file_path}")
        with file_path.open('r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'query-(\d+)\.cpp', f"{file_path}")
            if match:
                queryId = match.group(1)
                content += "\n\n" + code.get(queryId)['code']

        with open('tests/main.cpp', 'w', encoding='utf-8') as f:
            f.write(content)

        try:
            result = subprocess.run(
    ["bash", "-c", "g++ -fopenmp tests/main.cpp -o tests/main && ./tests/main"],
    timeout=10
)
            exit_code = result.returncode
            print(f"exit code: {exit_code}")
            print("\n\n")
            if exit_code == 0:
                success += 1
            else:
                fail += 1
        except subprocess.TimeoutExpired:
            print("Process timed out.")
            fail += 1

    print(f"success: {success}, fail: {fail}")
