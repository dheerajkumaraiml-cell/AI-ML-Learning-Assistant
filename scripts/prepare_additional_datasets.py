import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# FILES
# ============================================================

DATASET_2 = Path("dataset/raw/dataset_2.jsonl")
DATASET_3 = Path("dataset/raw/dataset_3.jsonl")
DATASET_4 = Path("dataset/raw/dataset_4.jsonl")

OUTPUT_DIR = Path("dataset/additional")


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# NORMALIZE QUESTION FOR DUPLICATE CHECKING
# ============================================================

def normalize_question(question):

    question = clean_text(question)

    question = question.lower()

    question = re.sub(
        r"[^\w\s]",
        "",
        question
    )

    question = re.sub(
        r"\s+",
        " ",
        question
    )

    return question.strip()


# ============================================================
# DATASET 2
# Python / Pandas
# ============================================================

def prepare_dataset_2():

    input_file = DATASET_2

    output_file = OUTPUT_DIR / "python_pandas_cleaned.jsonl"

    seen_questions = set()

    total = 0
    kept = 0
    duplicates = 0
    empty = 0

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as infile, open(
        output_file,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

            except json.JSONDecodeError:
                continue

            total += 1

            question = clean_text(
                data.get("question", "")
            )

            answer = clean_text(
                data.get("answer", "")
            )

            if not question or not answer:

                empty += 1
                continue

            question_key = normalize_question(
                question
            )

            if question_key in seen_questions:

                duplicates += 1
                continue

            seen_questions.add(
                question_key
            )

            new_item = {
                "question": question,
                "answer": answer,
                "topic": "Python/Data Science",
                "category": "Python",
                "level": "Intermediate",
                "source": "dataset_2"
            }

            outfile.write(
                json.dumps(
                    new_item,
                    ensure_ascii=False
                ) + "\n"
            )

            kept += 1

    return {
        "name": "Dataset 2 - Python/Pandas",
        "total": total,
        "kept": kept,
        "duplicates": duplicates,
        "empty": empty
    }


# ============================================================
# DATASET 3
# ML Evaluation
# ============================================================

def prepare_dataset_3():

    input_file = DATASET_3

    output_file = OUTPUT_DIR / "ml_evaluation_cleaned.jsonl"

    seen_questions = set()

    total = 0
    kept = 0
    duplicates = 0
    empty = 0

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as infile, open(
        output_file,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

            except json.JSONDecodeError:
                continue

            total += 1

            question = clean_text(
                data.get("question", "")
            )

            answer = clean_text(
                data.get("answer", "")
            )

            if not question or not answer:

                empty += 1
                continue

            question_key = normalize_question(
                question
            )

            if question_key in seen_questions:

                duplicates += 1
                continue

            seen_questions.add(
                question_key
            )

            new_item = {
                "question": question,
                "answer": answer,
                "topic": "Model Evaluation",
                "category": "Machine Learning",
                "level": "Intermediate",
                "source": "dataset_3"
            }

            outfile.write(
                json.dumps(
                    new_item,
                    ensure_ascii=False
                ) + "\n"
            )

            kept += 1

    return {
        "name": "Dataset 3 - ML Evaluation",
        "total": total,
        "kept": kept,
        "duplicates": duplicates,
        "empty": empty
    }


# ============================================================
# DATASET 4
# Coding / Debugging
# ============================================================

def prepare_dataset_4():

    input_file = DATASET_4

    output_file = OUTPUT_DIR / "coding_debugging_cleaned.jsonl"

    seen_questions = set()

    total = 0
    kept = 0
    duplicates = 0
    empty = 0
    revised_used = 0

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as infile, open(
        output_file,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

            except json.JSONDecodeError:
                continue

            total += 1

            # ------------------------------------------------
            # Dataset 4 uses "instruction", NOT "question"
            # ------------------------------------------------

            question = clean_text(
                data.get("instruction", "")
            )

            # ------------------------------------------------
            # Prefer revised_answer when available
            # ------------------------------------------------

            revised_answer = clean_text(
                data.get("revised_answer", "")
            )

            normal_answer = clean_text(
                data.get("answer", "")
            )

            if revised_answer:

                answer = revised_answer
                revised_used += 1

            else:

                answer = normal_answer

            # ------------------------------------------------
            # Required fields
            # ------------------------------------------------

            if not question or not answer:

                empty += 1
                continue

            # ------------------------------------------------
            # Duplicate detection
            # ------------------------------------------------

            question_key = normalize_question(
                question
            )

            if question_key in seen_questions:

                duplicates += 1
                continue

            seen_questions.add(
                question_key
            )

            # ------------------------------------------------
            # Final clean record
            # ------------------------------------------------

            new_item = {
                "question": question,
                "answer": answer,
                "topic": "Coding/Debugging",
                "category": "Coding/Debugging",
                "level": "Intermediate",
                "source": "dataset_4"
            }

            outfile.write(
                json.dumps(
                    new_item,
                    ensure_ascii=False
                ) + "\n"
            )

            kept += 1

    return {
        "name": "Dataset 4 - Coding/Debugging",
        "total": total,
        "kept": kept,
        "duplicates": duplicates,
        "empty": empty,
        "revised_answers_used": revised_used
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 65)
    print("PREPARING ADDITIONAL DATASETS")
    print("=" * 65)

    results = []

    # Dataset 2
    print("\nProcessing Dataset 2...")
    results.append(
        prepare_dataset_2()
    )

    # Dataset 3
    print("Processing Dataset 3...")
    results.append(
        prepare_dataset_3()
    )

    # Dataset 4
    print("Processing Dataset 4...")
    results.append(
        prepare_dataset_4()
    )

    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "=" * 65)
    print("CLEANING REPORT")
    print("=" * 65)

    for result in results:

        print("\n" + result["name"])
        print("-" * 50)

        print(
            f"Total       : {result['total']}"
        )

        print(
            f"Kept        : {result['kept']}"
        )

        print(
            f"Duplicates  : {result['duplicates']}"
        )

        print(
            f"Empty       : {result['empty']}"
        )

        if "revised_answers_used" in result:

            print(
                f"Revised answers used : "
                f"{result['revised_answers_used']}"
            )

    print("\n" + "=" * 65)
    print("FILES CREATED")
    print("=" * 65)

    print(
        OUTPUT_DIR /
        "python_pandas_cleaned.jsonl"
    )

    print(
        OUTPUT_DIR /
        "ml_evaluation_cleaned.jsonl"
    )

    print(
        OUTPUT_DIR /
        "coding_debugging_cleaned.jsonl"
    )

    print("\nDone!")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()