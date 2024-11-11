#dataset scaling prompts used in dataset_scaling.py
PROMPTS = [
            {"index": 0, "text": "Generate 10 OpenMP performance optimization multiple choice questions based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: Question :<generated question>, Options:<Four options A, B,C and D with one correct answer>, Answer: Correct answer to the generated question 'A', 'B', 'C' or 'D'>"},

            {"index": 1, "text": "Generate 10  multiple choice questions about advanced OpenMP performance optimization concepts based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: Question :<generated question>, Options:<Four options A, B,C and D with one correct answer>, Answer: Correct answer to the generated question 'A', 'B', 'C' or 'D'>"},
            {"index": 2, "text": "Generate 10 OpenMP performance optimization Yes/No questions based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: <generated question> Answer: <Correct answer to the generated question 'Yes' or 'No'>"},

            {"index": 3, "text": "Generate 10 Yes/No questions about advanced OpenMP performance optimization concepts based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question> Answer: <Correct answer to the generated question 'Yes' or 'No'>"},

            {"index":4, "text": "Generate 10 open-ended OpenMP performance optimization questions based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question> Answer: <Correct answer to the generated question>"},

            {"index": 5, "text": "Generate 10 open-ended questions about advanced OpenMP performance optimization concepts based on the given code snippet. The generated questions should be in the form of a list of json objects containing the following fields: Question: <generated question>  Answer: <Correct answer to the generated question>"}
            ]


EVALUATION_PROMPTS = {
    # MCQA prompts
    "MCQA": {
        "index": 0,
        "standard": ("Answer the following multiple choice question about openmp performance optimization. "
                     "The output should be the correct answer restricted to a single option letter 'A', 'B', 'C' or 'D' from the four options."),

        "cot": ("The following is a multiple choice question about openmp performance optimization. "
                "Solve it in a step-by-step fashion, starting by summarizing the available information. "
                "Output a single option from the four options as the final answer. The output format should be restricted to a json object with the following two fields: "
                "{'Explanation': <containing step-by-step reasoning>, 'Answer':<containing correct option letter>}.")
    },

    # YES/NO prompts
    "YES_NO": {
        "index": 1,
        "standard": ("The following is a Yes/No question about openmp performance optimization. "
                     "Output a 'Yes' or 'No' as the final answer."),
        "cot": ("The following is a Yes/No question about openmp performance optimization. "
                "Answer it in a step-by-step fashion, starting by summarizing the available information. "
                "Output a 'Yes' or 'No' as the final answer. The output should be in json format with the following fields: "
                "{'Explanation': <containing step-by-step reasoning>, 'Answer':'Yes' or 'No'}.")
    },

    # Open-ended prompts
    "OPEN_ENDED": {
        "index": 2,
        "text": "You are an openmp performance optimization expert. Provide useful, complete, and logically correct answers to performance optimization questions based on the given code samples.",
        "code": "You are an openmp performance optimization expert. Provide complete, syntactically and semantically correct answers to performance optimization questions based on the given code samples.",
    }
}

