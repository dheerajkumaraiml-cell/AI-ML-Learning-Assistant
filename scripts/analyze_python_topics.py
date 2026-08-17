import json
import re
from collections import Counter
from pathlib import Path


INPUT_FILE = Path(
    "dataset/additional/coding_debugging_cleaned.jsonl"
)


TOPICS = {
    "Python Basics": [
        "python",
        "variable",
        "data type",
        "string",
        "list",
        "tuple",
        "set",
        "dictionary",
        "dict",
        "loop",
        "for loop",
        "while loop",
        "if statement",
        "conditional"
    ],

    "Functions": [
        "function",
        "def ",
        "lambda",
        "recursion",
        "parameter",
        "argument"
    ],

    "OOP": [
        "class",
        "object",
        "inheritance",
        "polymorphism",
        "encapsulation",
        "abstraction",
        "constructor",
        "self",
        "method"
    ],

    "Exception Handling": [
        "exception",
        "try",
        "except",
        "finally",
        "raise",
        "error handling"
    ],

    "File Handling": [
        "file handling",
        "open(",
        "read file",
        "write file",
        "csv file",
        "json file"
    ],

    "NumPy": [
        "numpy",
        "np.",
        "ndarray",
        "array"
    ],

    "Pandas": [
        "pandas",
        "pd.",
        "dataframe",
        "series",
        "groupby",
        "merge",
        "concat",
        "pivot"
    ],

    "Machine Learning": [
        "machine learning",
        "scikit-learn",
        "sklearn",
        "train_test_split",
        "linear regression",
        "logistic regression",
        "random forest",
        "decision tree",
        "classification",
        "regression",
        "model training"
    ],

    "Algorithms": [
        "algorithm",
        "sorting",
        "searching",
        "binary search",
        "linear search",
        "merge sort",
        "quick sort",
        "bubble sort",
        "dynamic programming",
        "backtracking",
        "graph"
    ],

    "Data Structures": [
        "stack",
        "queue",
        "linked list",
        "tree",
        "binary tree",
        "heap",
        "hash table",
        "dictionary",
        "array"
    ],

    "Debugging": [
        "debug",
        "debugging",
        "fix this error",
        "error",
        "exception",
        "traceback",
        "bug",
        "not working",
        "why does this code"
    ],

    "Web/API": [
        "flask",
        "fastapi",
        "api",
        "rest api",
        "django",
        "http",
        "endpoint",
        "request",
        "response"
    ]
}


def normalize(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def analyze():

    counts = Counter()

    total = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

            except json.JSONDecodeError:
                continue

            total += 1

            question = normalize(
                data.get("question", "")
            )

            answer = normalize(
                data.get("answer", "")
            )

            combined = question + " " + answer

            matched = False

            for topic, keywords in TOPICS.items():

                for keyword in keywords:

                    if keyword.lower() in combined:

                        counts[topic] += 1

                        matched = True

                        break

            if not matched:

                counts["Other"] += 1

    print("\n" + "=" * 60)
    print("PYTHON DATASET TOPIC ANALYSIS")
    print("=" * 60)

    print(f"\nTotal examples: {total}")

    print("\nTOPIC DISTRIBUTION")
    print("-" * 60)

    for topic, count in counts.most_common():

        percentage = (
            count / total
        ) * 100

        print(
            f"{topic:<25}"
            f"{count:>7}"
            f" ({percentage:.2f}%)"
        )


if __name__ == "__main__":
    analyze()