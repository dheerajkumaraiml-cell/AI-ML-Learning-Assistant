import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# INPUT FILES
# ============================================================

INPUT_FILES = [

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
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = Path(
    "dataset/merged_dataset.jsonl"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Remove excessive spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# NORMALIZE QUESTION
# Used only for duplicate detection
# ============================================================

def normalize_question(question):

    question = clean_text(
        question
    ).lower()

    # Remove punctuation
    question = re.sub(
        r"[^\w\s]",
        " ",
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
# NORMALIZE LEVEL
# ============================================================

def normalize_level(level):

    level = clean_text(
        level
    ).lower()

    level_mapping = {

        "beginner": "Beginner",

        "intermediate": "Intermediate",

        "skilled": "Skilled",

        "advanced": "Advanced",

        "expert": "Expert",

        "exploration policies": "Advanced"
    }

    return level_mapping.get(
        level,
        ""
    )


# ============================================================
# NORMALIZE CATEGORY
# ============================================================

def normalize_category(category):

    category = clean_text(
        category
    )

    if not category:

        return "Unknown"

    return category


# ============================================================
# NORMALIZE TOPIC
# ============================================================

def normalize_topic(topic):

    topic = clean_text(
        topic
    )

    if not topic:

        return "General"

    return topic


# ============================================================
# LOAD DATASET
# ============================================================

def load_file(file_path):

    records = []

    invalid_json = 0

    empty_question = 0

    empty_answer = 0

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1
            ):

                line = line.strip()

                if not line:

                    continue

                # ------------------------------------------------
                # JSON validation
                # ------------------------------------------------

                try:

                    item = json.loads(
                        line
                    )

                except json.JSONDecodeError:

                    invalid_json += 1

                    continue

                # ------------------------------------------------
                # Question
                # ------------------------------------------------

                question = clean_text(
                    item.get(
                        "question",
                        ""
                    )
                )

                # ------------------------------------------------
                # Answer
                # ------------------------------------------------

                answer = clean_text(
                    item.get(
                        "answer",
                        ""
                    )
                )

                # ------------------------------------------------
                # Empty question
                # ------------------------------------------------

                if not question:

                    empty_question += 1

                    continue

                # ------------------------------------------------
                # Empty answer
                # ------------------------------------------------

                if not answer:

                    empty_answer += 1

                    continue

                # ------------------------------------------------
                # Category
                # ------------------------------------------------

                category = normalize_category(
                    item.get(
                        "category",
                        ""
                    )
                )

                # ------------------------------------------------
                # Topic
                # ------------------------------------------------

                topic = normalize_topic(
                    item.get(
                        "topic",
                        ""
                    )
                )

                # ------------------------------------------------
                # Level
                # ------------------------------------------------

                level = normalize_level(
                    item.get(
                        "level",
                        ""
                    )
                )

                # ------------------------------------------------
                # Source
                # ------------------------------------------------

                record = {

                    "question": question,

                    "answer": answer,

                    "topic": topic,

                    "category": category,

                    "level": level,

                    "source": file_path.name
                }

                records.append(
                    record
                )

    except FileNotFoundError:

        print(
            f"\n❌ FILE NOT FOUND:"
        )

        print(
            file_path
        )

        return (
            [],
            0,
            0,
            0
        )

    return (
        records,
        invalid_json,
        empty_question,
        empty_answer
    )


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

def remove_duplicates(records):

    unique_records = []

    seen_questions = set()

    duplicate_count = 0

    for record in records:

        key = normalize_question(
            record["question"]
        )

        if not key:

            continue

        if key in seen_questions:

            duplicate_count += 1

            continue

        seen_questions.add(
            key
        )

        unique_records.append(
            record
        )

    return (
        unique_records,
        duplicate_count
    )


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(records):

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
                )
                + "\n"
            )


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

def print_distribution(records):

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    categories = Counter()

    for record in records:

        categories[
            record["category"]
        ] += 1

    print(
        "\n" + "=" * 65
    )

    print(
        "CATEGORY DISTRIBUTION"
    )

    print(
        "=" * 65
    )

    total = len(records)

    for category, count in categories.most_common():

        percentage = (
            count / total
        ) * 100

        print(
            f"{category:<25}"
            f"{count:>7}"
            f" ({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # LEVEL
    # --------------------------------------------------------

    levels = Counter()

    for record in records:

        level = record["level"]

        if level:

            levels[
                level
            ] += 1

    print(
        "\n" + "=" * 65
    )

    print(
        "LEVEL DISTRIBUTION"
    )

    print(
        "=" * 65
    )

    for level, count in levels.most_common():

        percentage = (
            count / total
        ) * 100

        print(
            f"{level:<25}"
            f"{count:>7}"
            f" ({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    sources = Counter()

    for record in records:

        sources[
            record["source"]
        ] += 1

    print(
        "\n" + "=" * 65
    )

    print(
        "SOURCE DISTRIBUTION"
    )

    print(
        "=" * 65
    )

    for source, count in sources.most_common():

        percentage = (
            count / total
        ) * 100

        print(
            f"{source:<45}"
            f"{count:>7}"
            f" ({percentage:.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 65
    )

    print(
        "MERGING DATASETS"
    )

    print(
        "=" * 65
    )

    all_records = []

    total_invalid = 0

    total_empty_questions = 0

    total_empty_answers = 0

    # ========================================================
    # LOAD ALL FILES
    # ========================================================

    for file_path in INPUT_FILES:

        print(
            f"\nProcessing:"
        )

        print(
            file_path
        )

        (
            records,
            invalid,
            empty_questions,
            empty_answers
        ) = load_file(
            file_path
        )

        print(
            f"Valid records : "
            f"{len(records)}"
        )

        print(
            f"Invalid JSON  : "
            f"{invalid}"
        )

        print(
            f"Empty question: "
            f"{empty_questions}"
        )

        print(
            f"Empty answer  : "
            f"{empty_answers}"
        )

        all_records.extend(
            records
        )

        total_invalid += invalid

        total_empty_questions += (
            empty_questions
        )

        total_empty_answers += (
            empty_answers
        )

    # ========================================================
    # RAW TOTAL
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "RAW MERGED DATA"
    )

    print(
        "=" * 65
    )

    print(
        f"\nTotal valid records: "
        f"{len(all_records)}"
    )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    print(
        "\nRemoving exact duplicate questions..."
    )

    (
        unique_records,
        duplicates
    ) = remove_duplicates(
        all_records
    )

    print(
        f"Duplicate records removed: "
        f"{duplicates}"
    )

    print(
        f"Final unique records: "
        f"{len(unique_records)}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_jsonl(
        unique_records
    )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    print_distribution(
        unique_records
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "MERGE COMPLETE"
    )

    print(
        "=" * 65
    )

    print(
        f"\nRaw records              : "
        f"{len(all_records)}"
    )

    print(
        f"Invalid JSON             : "
        f"{total_invalid}"
    )

    print(
        f"Empty questions removed  : "
        f"{total_empty_questions}"
    )

    print(
        f"Empty answers removed    : "
        f"{total_empty_answers}"
    )

    print(
        f"Duplicates removed       : "
        f"{duplicates}"
    )

    print(
        f"Final unique records     : "
        f"{len(unique_records)}"
    )

    if len(all_records) > 0:

        duplicate_percentage = (
            duplicates
            / len(all_records)
        ) * 100

        print(
            f"Duplicate percentage     : "
            f"{duplicate_percentage:.2f}%"
        )

    print(
        "\nOutput file:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nExisting source files were NOT modified."
    )

    print(
        "\nDone!"
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()