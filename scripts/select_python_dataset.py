import json
import re
import random
from pathlib import Path
from collections import defaultdict, Counter


# ============================================================
# FILE PATHS
# ============================================================

DATASET_2 = Path(
    "dataset/additional/python_pandas_cleaned.jsonl"
)

DATASET_4 = Path(
    "dataset/additional/coding_debugging_cleaned.jsonl"
)

OUTPUT_FILE = Path(
    "dataset/additional/python_balanced_1000.jsonl"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_TOTAL = 1000

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# TOPIC KEYWORDS
# ============================================================

TOPIC_KEYWORDS = {

    "Algorithms": [
        "algorithm",
        "algorithms",
        "sorting",
        "searching",
        "binary search",
        "linear search",
        "merge sort",
        "quick sort",
        "bubble sort",
        "insertion sort",
        "selection sort",
        "heap sort",
        "recursion",
        "dynamic programming",
        "backtracking",
        "greedy",
        "graph",
        "graphs",
        "shortest path",
        "dijkstra",
        "bfs",
        "dfs",
        "breadth first search",
        "depth first search"
    ],

    "NumPy": [
        "numpy",
        "np.",
        "ndarray",
        "numpy array",
        "numpy arrays",
        "np.array",
        "np.zeros",
        "np.ones",
        "np.reshape",
        "np.mean",
        "np.sum",
        "np.random",
        "np.arange",
        "np.linspace"
    ],

    "Debugging": [
        "debug",
        "debugging",
        "bug",
        "error",
        "errors",
        "exception",
        "exceptions",
        "traceback",
        "fix this error",
        "fix the error",
        "not working",
        "why does this code",
        "error message",
        "runtime error",
        "syntax error",
        "typeerror",
        "valueerror",
        "indexerror",
        "keyerror",
        "attributeerror",
        "nameerror",
        "zerodivisionerror"
    ],

    "Web/API": [
        "flask",
        "fastapi",
        "django",
        "api",
        "rest api",
        "restful api",
        "endpoint",
        "http",
        "request",
        "response",
        "web application",
        "web app",
        "route",
        "routing",
        "json api",
        "api endpoint"
    ],

    "Pandas": [
        "pandas",
        "pd.",
        "dataframe",
        "data frame",
        "series",
        "groupby",
        "group by",
        "merge",
        "concat",
        "pivot",
        "pivot_table",
        "read_csv",
        "to_csv",
        "dropna",
        "fillna",
        "loc",
        "iloc",
        "dataframe manipulation"
    ],

    "Machine Learning": [
        "machine learning",
        "scikit-learn",
        "scikit learn",
        "sklearn",
        "train_test_split",
        "linear regression",
        "logistic regression",
        "random forest",
        "decision tree",
        "classification",
        "regression",
        "model training",
        "model prediction",
        "fit model",
        "predict",
        "train model",
        "ml model"
    ],

    "File Handling": [
        "file handling",
        "file handling in python",
        "open(",
        "read file",
        "write file",
        "file",
        "csv file",
        "json file",
        "text file",
        "read",
        "write",
        "append",
        "file path",
        "directory",
        "folder"
    ],

    "Python Basics": [
        "python",
        "variable",
        "variables",
        "data type",
        "data types",
        "string",
        "strings",
        "list",
        "lists",
        "tuple",
        "tuples",
        "set",
        "sets",
        "dictionary",
        "dictionary",
        "dict",
        "loop",
        "loops",
        "for loop",
        "while loop",
        "if statement",
        "conditional",
        "conditional statement",
        "comprehension",
        "list comprehension"
    ],

    "Functions": [
        "function",
        "functions",
        "def ",
        "lambda",
        "parameter",
        "parameters",
        "argument",
        "arguments",
        "return value",
        "recursive function",
        "function definition"
    ],

    "OOP": [
        "class",
        "classes",
        "object",
        "objects",
        "inheritance",
        "polymorphism",
        "encapsulation",
        "abstraction",
        "constructor",
        "self",
        "method",
        "methods",
        "object-oriented",
        "object oriented",
        "oop"
    ]
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# NORMALIZE QUESTION
# Used for duplicate detection
# ============================================================

def normalize_question(question):

    question = clean_text(
        question
    ).lower()

    # Remove punctuation
    question = re.sub(
        r"[^\w\s]",
        "",
        question
    )

    # Remove extra spaces
    question = re.sub(
        r"\s+",
        " ",
        question
    )

    return question.strip()


# ============================================================
# DETECT TOPIC
# ============================================================

def detect_topic(question, answer):

    combined_text = (
        clean_text(question)
        + " "
        + clean_text(answer)
    ).lower()

    topic_scores = defaultdict(int)

    for topic, keywords in TOPIC_KEYWORDS.items():

        for keyword in keywords:

            keyword = keyword.lower()

            if keyword in combined_text:

                topic_scores[topic] += 1

    # If nothing matches
    if not topic_scores:

        return "Python Basics"

    # Return topic having highest score
    return max(
        topic_scores,
        key=topic_scores.get
    )


# ============================================================
# LOAD DATASET 2
# Python / Pandas
# ============================================================

def load_dataset_2():

    records = []

    if not DATASET_2.exists():

        print(
            f"WARNING: File not found: {DATASET_2}"
        )

        return records

    with open(
        DATASET_2,
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

            question = clean_text(
                data.get(
                    "question",
                    ""
                )
            )

            answer = clean_text(
                data.get(
                    "answer",
                    ""
                )
            )

            if not question or not answer:

                continue

            records.append({

                "question": question,

                "answer": answer,

                "source": "dataset_2"

            })

    return records


# ============================================================
# LOAD DATASET 4
# Python Coding / Debugging
# ============================================================

def load_dataset_4():

    records = []

    if not DATASET_4.exists():

        print(
            f"WARNING: File not found: {DATASET_4}"
        )

        return records

    with open(
        DATASET_4,
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

            question = clean_text(
                data.get(
                    "question",
                    ""
                )
            )

            answer = clean_text(
                data.get(
                    "answer",
                    ""
                )
            )

            if not question or not answer:

                continue

            records.append({

                "question": question,

                "answer": answer,

                "source": "dataset_4"

            })

    return records


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(records):

    unique_records = []

    seen = set()

    duplicates = 0

    for record in records:

        key = normalize_question(
            record["question"]
        )

        if not key:

            continue

        if key in seen:

            duplicates += 1

            continue

        seen.add(key)

        unique_records.append(
            record
        )

    return unique_records, duplicates


# ============================================================
# QUALITY FILTER
# ============================================================

def quality_filter(records):

    filtered = []

    removed = 0

    for record in records:

        question = record["question"]

        answer = record["answer"]

        # Very short question
        if len(question) < 10:

            removed += 1

            continue

        # Very short answer
        if len(answer) < 30:

            removed += 1

            continue

        # Extremely long answer
        if len(answer) > 10000:

            removed += 1

            continue

        filtered.append(
            record
        )

    return filtered, removed


# ============================================================
# GROUP RECORDS BY TOPIC
# ============================================================

def group_by_topic(records):

    groups = defaultdict(list)

    for record in records:

        topic = detect_topic(
            record["question"],
            record["answer"]
        )

        record["topic"] = topic

        record["category"] = "Python"

        groups[topic].append(
            record
        )

    return groups


# ============================================================
# TARGETS
# ============================================================

TOPIC_TARGETS = {

    "Algorithms": 180,

    "NumPy": 160,

    "Debugging": 180,

    "Web/API": 140,

    "Pandas": 140,

    "Machine Learning": 100,

    "File Handling": 50,

    "Python Basics": 20,

    "Functions": 10,

    "OOP": 10
}


# ============================================================
# SELECT BALANCED DATA
# ============================================================

def select_balanced(groups):

    selected = []

    remaining = []

    print("\n" + "=" * 60)
    print("TOPIC-WISE SELECTION")
    print("=" * 60)

    for topic, target in TOPIC_TARGETS.items():

        available = list(
            groups.get(
                topic,
                []
            )
        )

        random.shuffle(
            available
        )

        selected_count = min(
            target,
            len(available)
        )

        chosen = available[
            :selected_count
        ]

        selected.extend(
            chosen
        )

        print(
            f"{topic:<25}"
            f"Available: {len(available):>5}"
            f" | Selected: {selected_count:>4}"
        )

    # ========================================================
    # FILL REMAINING SLOTS
    # ========================================================

    if len(selected) < TARGET_TOTAL:

        selected_questions = {

            normalize_question(
                record["question"]
            )

            for record in selected
        }

        for topic_records in groups.values():

            for record in topic_records:

                key = normalize_question(
                    record["question"]
                )

                if key not in selected_questions:

                    remaining.append(
                        record
                    )

        random.shuffle(
            remaining
        )

        needed = (
            TARGET_TOTAL
            - len(selected)
        )

        selected.extend(
            remaining[
                :needed
            ]
        )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(selected) > TARGET_TOTAL:

        random.shuffle(
            selected
        )

        selected = selected[
            :TARGET_TOTAL
        ]

    return selected


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(records):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


# ============================================================
# FINAL TOPIC DISTRIBUTION
# ============================================================

def show_final_distribution(records):

    final_topics = Counter()

    for record in records:

        final_topics[
            record["topic"]
        ] += 1

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL TOPIC DISTRIBUTION"
    )

    print(
        "=" * 60
    )

    total = len(records)

    for topic, count in final_topics.most_common():

        percentage = (
            count / total
        ) * 100

        print(
            f"{topic:<25}"
            f"{count:>5}"
            f" ({percentage:.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 60
    )

    print(
        "PYTHON DATASET SELECTION"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # LOAD DATASET 2
    # ========================================================

    print(
        "\nLoading Dataset 2..."
    )

    dataset_2 = load_dataset_2()

    print(
        f"Dataset 2 loaded: "
        f"{len(dataset_2)}"
    )

    # ========================================================
    # LOAD DATASET 4
    # ========================================================

    print(
        "\nLoading Dataset 4..."
    )

    dataset_4 = load_dataset_4()

    print(
        f"Dataset 4 loaded: "
        f"{len(dataset_4)}"
    )

    # ========================================================
    # COMBINE
    # ========================================================

    combined = (
        dataset_2
        + dataset_4
    )

    print(
        f"\nCombined records: "
        f"{len(combined)}"
    )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    combined, duplicates = (
        remove_duplicates(
            combined
        )
    )

    print(
        f"Duplicates removed: "
        f"{duplicates}"
    )

    print(
        f"After deduplication: "
        f"{len(combined)}"
    )

    # ========================================================
    # QUALITY FILTER
    # ========================================================

    combined, removed = (
        quality_filter(
            combined
        )
    )

    print(
        f"Low-quality records removed: "
        f"{removed}"
    )

    print(
        f"After quality filtering: "
        f"{len(combined)}"
    )

    # ========================================================
    # GROUP BY TOPIC
    # ========================================================

    groups = group_by_topic(
        combined
    )

    print(
        "\nTopic availability:"
    )

    for topic, records in sorted(
        groups.items(),
        key=lambda x: len(x[1]),
        reverse=True
    ):

        print(
            f"{topic:<25}"
            f"{len(records):>6}"
        )

    # ========================================================
    # SELECT BALANCED DATA
    # ========================================================

    selected = select_balanced(
        groups
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_dataset(
        selected
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL PYTHON DATASET"
    )

    print(
        "=" * 60
    )

    print(
        f"\nTotal selected: "
        f"{len(selected)}"
    )

    print(
        "\nOutput file:"
    )

    print(
        OUTPUT_FILE
    )

    # ========================================================
    # FINAL TOPIC DISTRIBUTION
    # ========================================================

    show_final_distribution(
        selected
    )

    print(
        "\nDone!"
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()