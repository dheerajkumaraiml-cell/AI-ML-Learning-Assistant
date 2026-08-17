import json
from pathlib import Path
from collections import Counter


# ============================================================
# DATASET FILES
# ============================================================

DATASETS = [
    Path("dataset/raw/concise_questions_with_answers.jsonl"),
    Path("dataset/raw/dataset_2.jsonl"),
    Path("dataset/raw/dataset_3.jsonl"),
    Path("dataset/raw/dataset_4.jsonl"),
]


# ============================================================
# ANALYZE ONE DATASET
# ============================================================

def analyze_dataset(file_path):

    total = 0
    valid = 0
    invalid = 0
    empty_question = 0
    empty_answer = 0

    questions = set()

    duplicate_questions = 0

    fields = Counter()

    topics = Counter()
    levels = Counter()

    answer_lengths = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            total += 1

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

            except json.JSONDecodeError:
                invalid += 1
                continue

            valid += 1

            # ------------------------------------------------
            # Fields
            # ------------------------------------------------

            for field in data.keys():
                fields[field] += 1

            # ------------------------------------------------
            # Question
            # ------------------------------------------------

            question = data.get("question", "")

            if not isinstance(question, str):
                question = ""

            question = question.strip()

            if not question:
                empty_question += 1

            # ------------------------------------------------
            # Duplicate question
            # ------------------------------------------------

            normalized_question = question.lower()

            if normalized_question:

                if normalized_question in questions:
                    duplicate_questions += 1

                else:
                    questions.add(normalized_question)

            # ------------------------------------------------
            # Answer
            # ------------------------------------------------

            answer = data.get("answer", "")

            if not isinstance(answer, str):
                answer = ""

            answer = answer.strip()

            if not answer:
                empty_answer += 1

            else:
                answer_lengths.append(len(answer))

            # ------------------------------------------------
            # Topic
            # ------------------------------------------------

            topic = data.get("topic", "")

            if topic:
                topics[str(topic)] += 1

            # ------------------------------------------------
            # Level
            # ------------------------------------------------

            level = data.get("level", "")

            if level:
                levels[str(level)] += 1

    # --------------------------------------------------------
    # Average answer length
    # --------------------------------------------------------

    if answer_lengths:

        average_answer_length = (
            sum(answer_lengths)
            / len(answer_lengths)
        )

    else:

        average_answer_length = 0

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "empty_question": empty_question,
        "empty_answer": empty_answer,
        "duplicate_questions": duplicate_questions,
        "fields": fields,
        "topics": topics,
        "levels": levels,
        "average_answer_length": average_answer_length
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("DATASET COMPARISON")
    print("=" * 70)

    for dataset in DATASETS:

        print("\n")
        print("=" * 70)
        print(f"FILE: {dataset.name}")
        print("=" * 70)

        if not dataset.exists():

            print("❌ File not found")

            continue

        result = analyze_dataset(dataset)

        print(f"\nTotal rows           : {result['total']}")
        print(f"Valid JSON           : {result['valid']}")
        print(f"Invalid JSON         : {result['invalid']}")

        print(
            f"Empty questions      : "
            f"{result['empty_question']}"
        )

        print(
            f"Empty answers        : "
            f"{result['empty_answer']}"
        )

        print(
            f"Duplicate questions  : "
            f"{result['duplicate_questions']}"
        )

        print(
            f"Average answer chars : "
            f"{result['average_answer_length']:.0f}"
        )

        # ----------------------------------------------------
        # Fields
        # ----------------------------------------------------

        print("\nFields:")

        for field, count in result["fields"].items():

            print(
                f"  {field:<20} {count}"
            )

        # ----------------------------------------------------
        # Topics
        # ----------------------------------------------------

        if result["topics"]:

            print("\nTop Topics:")

            for topic, count in result["topics"].most_common(10):

                print(
                    f"  {topic:<35} {count}"
                )

        # ----------------------------------------------------
        # Levels
        # ----------------------------------------------------

        if result["levels"]:

            print("\nLevels:")

            for level, count in result["levels"].most_common():

                print(
                    f"  {level:<20} {count}"
                )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()