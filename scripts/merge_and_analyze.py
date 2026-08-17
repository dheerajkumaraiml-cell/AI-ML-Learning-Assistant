import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# INPUT DATASETS
# ============================================================

FILES = [
    Path("dataset/normalized_dataset.jsonl"),

    Path(
        "dataset/additional/"
        "python_balanced_1000.jsonl"
    ),

    Path(
        "dataset/additional/"
        "ml_evaluation_cleaned.jsonl"
    ),

    Path(
        "dataset/additional/"
        "coding_debugging_cleaned.jsonl"
    )
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


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
# LOAD ONE DATASET
# ============================================================

def load_dataset(file_path):

    records = []

    invalid = 0

    empty_question = 0

    empty_answer = 0

    short_answer = 0

    try:

        with open(
            file_path,
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

                    invalid += 1

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

                if not question:

                    empty_question += 1

                    continue

                if not answer:

                    empty_answer += 1

                    continue

                if len(answer) < 30:

                    short_answer += 1

                records.append({

                    "question": question,

                    "answer": answer,

                    "topic": clean_text(
                        data.get(
                            "topic",
                            ""
                        )
                    ),

                    "category": clean_text(
                        data.get(
                            "category",
                            ""
                        )
                    ),

                    "level": clean_text(
                        data.get(
                            "level",
                            ""
                        )
                    ),

                    "source": file_path.name

                })

    except FileNotFoundError:

        print(
            f"\n❌ FILE NOT FOUND: {file_path}"
        )

    return (
        records,
        invalid,
        empty_question,
        empty_answer,
        short_answer
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "MERGE AND ANALYZE DATASETS"
    )

    print("=" * 70)

    all_records = []

    dataset_statistics = {}

    # ========================================================
    # READ ALL DATASETS
    # ========================================================

    for file_path in FILES:

        print(
            f"\nReading: {file_path}"
        )

        records, invalid, empty_q, empty_a, short_a = (
            load_dataset(file_path)
        )

        all_records.extend(records)

        dataset_statistics[
            file_path.name
        ] = {

            "loaded": len(records),

            "invalid": invalid,

            "empty_question": empty_q,

            "empty_answer": empty_a,

            "short_answer": short_a

        }

        print(
            f"Valid records : {len(records)}"
        )

        print(
            f"Invalid JSON  : {invalid}"
        )

        print(
            f"Empty question: {empty_q}"
        )

        print(
            f"Empty answer  : {empty_a}"
        )

        print(
            f"Short answers : {short_a}"
        )

    # ========================================================
    # TOTAL RECORDS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "TOTAL DATA"
    )

    print("=" * 70)

    print(
        f"\nTotal loaded records: "
        f"{len(all_records)}"
    )

    # ========================================================
    # DUPLICATE ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "DUPLICATE ANALYSIS"
    )

    print("=" * 70)

    question_counter = Counter()

    question_sources = {}

    for record in all_records:

        question_key = normalize_question(
            record["question"]
        )

        if not question_key:
            continue

        question_counter[
            question_key
        ] += 1

        if question_key not in question_sources:

            question_sources[
                question_key
            ] = []

        question_sources[
            question_key
        ].append(
            record["source"]
        )

    unique_questions = len(
        question_counter
    )

    duplicate_groups = sum(
        1
        for count in question_counter.values()
        if count > 1
    )

    duplicate_records = sum(
        count - 1
        for count in question_counter.values()
        if count > 1
    )

    print(
        f"\nUnique questions     : "
        f"{unique_questions}"
    )

    print(
        f"Duplicate groups     : "
        f"{duplicate_groups}"
    )

    print(
        f"Duplicate records    : "
        f"{duplicate_records}"
    )

    # ========================================================
    # TOP DUPLICATE QUESTIONS
    # ========================================================

    print("\nTop duplicate questions:")

    duplicate_items = [

        (
            question,
            count
        )

        for question, count
        in question_counter.items()

        if count > 1

    ]

    duplicate_items.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for question, count in duplicate_items[:20]:

        print(
            f"\n[{count} times] "
            f"{question[:150]}"
        )

        sources = question_sources[
            question
        ]

        print(
            "Sources: "
            + ", ".join(
                sources
            )
        )

    # ========================================================
    # CATEGORY DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "CATEGORY DISTRIBUTION"
    )

    print("=" * 70)

    categories = Counter()

    for record in all_records:

        category = record["category"]

        if not category:

            category = "Unknown"

        categories[
            category
        ] += 1

    for category, count in categories.most_common():

        percentage = (
            count
            / len(all_records)
            * 100
        )

        print(
            f"{category:<25}"
            f"{count:>7}"
            f" ({percentage:.2f}%)"
        )

    # ========================================================
    # TOPICS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "TOP TOPICS"
    )

    print("=" * 70)

    topics = Counter()

    for record in all_records:

        topic = record["topic"]

        if topic:

            topics[
                topic
            ] += 1

    for topic, count in topics.most_common(30):

        print(
            f"{topic:<40}"
            f"{count:>6}"
        )

    # ========================================================
    # LEVEL DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "LEVEL DISTRIBUTION"
    )

    print("=" * 70)

    levels = Counter()

    for record in all_records:

        level = record["level"]

        if level:

            levels[
                level
            ] += 1

    if levels:

        for level, count in levels.most_common():

            percentage = (
                count
                / len(all_records)
                * 100
            )

            print(
                f"{level:<25}"
                f"{count:>7}"
                f" ({percentage:.2f}%)"
            )

    else:

        print(
            "No level information available."
        )

    # ========================================================
    # SOURCE DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "SOURCE DISTRIBUTION"
    )

    print("=" * 70)

    sources = Counter(
        record["source"]
        for record in all_records
    )

    for source, count in sources.items():

        percentage = (
            count
            / len(all_records)
            * 100
        )

        print(
            f"{source:<45}"
            f"{count:>7}"
            f" ({percentage:.2f}%)"
        )

    # ========================================================
    # ANSWER LENGTH
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "ANSWER QUALITY STATISTICS"
    )

    print("=" * 70)

    answer_lengths = [

        len(
            record["answer"]
        )

        for record in all_records

    ]

    if answer_lengths:

        average_length = (
            sum(answer_lengths)
            / len(answer_lengths)
        )

        short_answers = sum(
            1
            for length
            in answer_lengths
            if length < 30
        )

        very_short_answers = sum(
            1
            for length
            in answer_lengths
            if length < 100
        )

        long_answers = sum(
            1
            for length
            in answer_lengths
            if length > 3000
        )

        print(
            f"\nAverage answer length : "
            f"{average_length:.0f} characters"
        )

        print(
            f"Answers < 30 chars   : "
            f"{short_answers}"
        )

        print(
            f"Answers < 100 chars  : "
            f"{very_short_answers}"
        )

        print(
            f"Answers > 3000 chars : "
            f"{long_answers}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "FINAL SUMMARY"
    )

    print("=" * 70)

    print(
        f"\nRaw combined records     : "
        f"{len(all_records)}"
    )

    print(
        f"Unique questions         : "
        f"{unique_questions}"
    )

    print(
        f"Duplicate records        : "
        f"{duplicate_records}"
    )

    print(
        f"Duplicate percentage     : "
        f"{(duplicate_records / len(all_records) * 100):.2f}%"
    )

    print(
        "\n⚠️ IMPORTANT:"
    )

    print(
        "This script ONLY analyzes the datasets."
    )

    print(
        "It does NOT delete or modify any existing file."
    )

    print(
        "No final dataset has been created yet."
    )

    print(
        "\nNext step will be decided from this report."
    )

    print(
        "\nDone!"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()