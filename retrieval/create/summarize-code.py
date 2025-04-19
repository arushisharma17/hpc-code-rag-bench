import json

import openai
from openai import OpenAI

client = OpenAI()

def generate_humaneval_doc_and_signature(cpp_function_code: str) -> str:
    prompt = f"""
Given the following C/C++ function, write a concise summary describing what the function does in natural language, following the HumanEval style. The summary should be written above the function signature. It should explain:
- The purpose of the function.
- Any algorithms or techniques used (e.g., parallelism, recursion, sorting).
- Key parameters (if any), and what the function returns.

Then, write the function signature only (without the implementation), using correct C/C++ syntax.

Format the output as:

 <Function purpose and behavior>

 Parameters:
 <param_name> – <description> (if any)

 Returns:
     <return value description>

<function_signature>;

Given function

{cpp_function_code}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful AI that explains code in HumanEval style."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def main():
    with open('need-to-summarize.json', 'r+') as f:
        rows = json.load(f)
        for row in rows:
            code = row['code']
            res = generate_humaneval_doc_and_signature(code)
            row['summarized_code'] = res
            print(res)
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()