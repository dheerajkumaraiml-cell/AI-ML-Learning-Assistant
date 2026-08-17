import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# INPUT FILE
# ============================================================

INPUT_FILE = Path(
    "dataset/merged_dataset.jsonl"
)


# ============================================================
# SETTINGS
# ============================================================

# These are ONLY for reporting.
# Nothing will be deleted automatically.

SHORT_QUESTION_LENGTH = 10

VERY_SHORT_ANSWER_LENGTH = 30

SHORT_ANSWER_LENGTH = 100

LONG_ANSWER_LENGTH = 3000

VERY_LONG_ANSWER_LENGTH = 10000


# ============================================================
# TEXT CLEANING
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


# ============================================================
# NORMALIZE QUESTION
# ============================================================

def normalize_question(question):

    question = clean_text(
        question
    ).lower()

    question = re.sub(
        r"[^\w\s]",
        " ",
        question
    )

    question = re.sub(
        r"\s+",
        " ",
        question
    )

    return question.strip()


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    records = []

    invalid_json = 0

    if not INPUT_FILE.exists():

        print(
            "\n❌ Dataset file not found:"
        )

        print(
            INPUT_FILE
        )

        return records, invalid_json

    with open(
        INPUT_FILE,
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

            try:

                item = json.loads(
                    line
                )

            except json.JSONDecodeError:

                invalid_json += 1

                continue

            records.append(
                item
            )

    return records, invalid_json


# ============================================================
# MAIN QUALITY ANALYSIS
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL DATASET QUALITY CHECK"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # LOAD
    # ========================================================

    records, invalid_json = (
        load_dataset()
    )

    total = len(records)

    print(
        f"\nDataset:"
    )

    print(
        INPUT_FILE
    )

    print(
        f"\nTotal valid records: "
        f"{total}"
    )

    print(
        f"Invalid JSON records: "
        f"{invalid_json}"
    )

    if total == 0:

        print(
            "\n❌ No valid records found."
        )

        return

    # ========================================================
    # COUNTERS
    # ========================================================

    empty_questions = 0

    empty_answers = 0

    short_questions = 0

    very_short_answers = 0

    short_answers = 0

    long_answers = 0

    very_long_answers = 0

    missing_topic = 0

    missing_category = 0

    missing_level = 0

    suspicious_records = 0

    duplicate_questions = 0

    question_counter = Counter()

    categories = Counter()

    topics = Counter()

    levels = Counter()

    sources = Counter()

    answer_lengths = []

    question_lengths = []

    problematic_records = []

    # ========================================================
    # CHECK EACH RECORD
    # ========================================================

    for index, record in enumerate(
        records,
        start=1
    ):

        question = clean_text(
            record.get(
                "question",
                ""
            )
        )

        answer = clean_text(
            record.get(
                "answer",
                ""
            )
        )

        topic = clean_text(
            record.get(
                "topic",
                ""
            )
        )

        category = clean_text(
            record.get(
                "category",
                ""
            )
        )

        level = clean_text(
            record.get(
                "level",
                ""
            )
        )

        source = clean_text(
            record.get(
                "source",
                ""
            )
        )

        # ----------------------------------------------------
        # QUESTION CHECK
        # ----------------------------------------------------

        if not question:

            empty_questions += 1

        else:

            question_length = len(
                question
            )

            question_lengths.append(
                question_length
            )

            if (
                question_length
                < SHORT_QUESTION_LENGTH
            ):

                short_questions += 1

                if len(
                    problematic_records
                ) < 20:

                    problematic_records.append({

                        "type":
                            "Short Question",

                        "index":
                            index,

                        "question":
                            question,

                        "answer":
                            answer[:300]
                    })

        # ----------------------------------------------------
        # ANSWER CHECK
        # ----------------------------------------------------

        if not answer:

            empty_answers += 1

        else:

            answer_length = len(
                answer
            )

            answer_lengths.append(
                answer_length
            )

            if (
                answer_length
                < VERY_SHORT_ANSWER_LENGTH
            ):

                very_short_answers += 1

            if (
                answer_length
                < SHORT_ANSWER_LENGTH
            ):

                short_answers += 1

            if (
                answer_length
                > LONG_ANSWER_LENGTH
            ):

                long_answers += 1

            if (
                answer_length
                > VERY_LONG_ANSWER_LENGTH
            ):

                very_long_answers += 1

        # ----------------------------------------------------
        # METADATA CHECK
        # ----------------------------------------------------

        if not topic:

            missing_topic += 1

        if not category:

            missing_category += 1

        if not level:

            missing_level += 1

        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        if category:

            categories[
                category
            ] += 1

        else:

            categories[
                "Unknown"
            ] += 1

        # ----------------------------------------------------
        # TOPIC
        # ----------------------------------------------------

        if topic:

            topics[
                topic
            ] += 1

        else:

            topics[
                "Unknown"
            ] += 1

        # ----------------------------------------------------
        # LEVEL
        # ----------------------------------------------------

        if level:

            levels[
                level
            ] += 1

        else:

            levels[
                "Unknown"
            ] += 1

        # ----------------------------------------------------
        # SOURCE
        # ----------------------------------------------------

        if source:

            sources[
                source
            ] += 1

        else:

            sources[
                "Unknown"
            ] += 1

        # ----------------------------------------------------
        # DUPLICATE QUESTION CHECK
        # ----------------------------------------------------

        normalized_q = normalize_question(
            question
        )

        if normalized_q:

            question_counter[
                normalized_q
            ] += 1

        # ----------------------------------------------------
        # SUSPICIOUS CONTENT
        # ----------------------------------------------------

        suspicious = False

        lower_question = question.lower()

        lower_answer = answer.lower()

        # Repeated meaningless answer
        if answer and len(
            set(
                lower_answer.split()
            )
        ) <= 3 and len(answer) > 50:

            suspicious = True

        # Common bad placeholders
        bad_phrases = [

            "i don't know",

            "dont know",

            "no answer",

            "not available",

            "n/a",

            "none",

            "null",

            "undefined"
        ]

        for phrase in bad_phrases:

            if (
                lower_answer.strip()
                == phrase
            ):

                suspicious = True

        if suspicious:

            suspicious_records += 1

            if len(
                problematic_records
            ) < 20:

                problematic_records.append({

                    "type":
                        "Suspicious Answer",

                    "index":
                        index,

                    "question":
                        question,

                    "answer":
                        answer[:300]
                })

    # ========================================================
    # DUPLICATE RESULTS
    # ========================================================

    duplicate_groups = sum(

        1

        for count
        in question_counter.values()

        if count > 1
    )

    duplicate_records = sum(

        count - 1

        for count
        in question_counter.values()

        if count > 1
    )

    unique_questions = len(
        question_counter
    )

    # ========================================================
    # ANSWER STATISTICS
    # ========================================================

    if answer_lengths:

        average_answer = (
            sum(answer_lengths)
            / len(answer_lengths)
        )

        minimum_answer = min(
            answer_lengths
        )

        maximum_answer = max(
            answer_lengths
        )

    else:

        average_answer = 0

        minimum_answer = 0

        maximum_answer = 0

    # ========================================================
    # QUESTION STATISTICS
    # ========================================================

    if question_lengths:

        average_question = (
            sum(question_lengths)
            / len(question_lengths)
        )

        minimum_question = min(
            question_lengths
        )

        maximum_question = max(
            question_lengths
        )

    else:

        average_question = 0

        minimum_question = 0

        maximum_question = 0

    # ========================================================
    # BASIC QUALITY REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "BASIC QUALITY CHECK"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal records             : "
        f"{total}"
    )

    print(
        f"Invalid JSON              : "
        f"{invalid_json}"
    )

    print(
        f"Empty questions           : "
        f"{empty_questions}"
    )

    print(
        f"Empty answers             : "
        f"{empty_answers}"
    )

    print(
        f"Short questions <10       : "
        f"{short_questions}"
    )

    print(
        f"Answers <30 chars         : "
        f"{very_short_answers}"
    )

    print(
        f"Answers <100 chars        : "
        f"{short_answers}"
    )

    print(
        f"Answers >3000 chars       : "
        f"{long_answers}"
    )

    print(
        f"Answers >10000 chars      : "
        f"{very_long_answers}"
    )

    # ========================================================
    # DUPLICATE REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "DUPLICATE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nUnique questions          : "
        f"{unique_questions}"
    )

    print(
        f"Duplicate groups          : "
        f"{duplicate_groups}"
    )

    print(
        f"Duplicate records         : "
        f"{duplicate_records}"
    )

    if total:

        print(
            f"Duplicate percentage      : "
            f"{duplicate_records / total * 100:.2f}%"
        )

    # ========================================================
    # METADATA REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "METADATA CHECK"
    )

    print(
        "=" * 70
    )

    print(
        f"\nMissing topic             : "
        f"{missing_topic}"
    )

    print(
        f"Missing category          : "
        f"{missing_category}"
    )

    print(
        f"Missing level             : "
        f"{missing_level}"
    )

    print(
        f"Suspicious records        : "
        f"{suspicious_records}"
    )

    # ========================================================
    # ANSWER LENGTH REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ANSWER LENGTH STATISTICS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAverage answer length    : "
        f"{average_answer:.0f} characters"
    )

    print(
        f"Minimum answer length    : "
        f"{minimum_answer} characters"
    )

    print(
        f"Maximum answer length    : "
        f"{maximum_answer} characters"
    )

    # ========================================================
    # QUESTION LENGTH REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "QUESTION LENGTH STATISTICS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAverage question length  : "
        f"{average_question:.0f} characters"
    )

    print(
        f"Minimum question length  : "
        f"{minimum_question} characters"
    )

    print(
        f"Maximum question length  : "
        f"{maximum_question} characters"
    )

    # ========================================================
    # CATEGORY DISTRIBUTION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CATEGORY DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    for category, count in categories.most_common():

        percentage = (
            count
            / total
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

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP 30 TOPICS"
    )

    print(
        "=" * 70
    )

    for topic, count in topics.most_common(30):

        print(
            f"{topic:<40}"
            f"{count:>7}"
        )

    # ========================================================
    # LEVEL DISTRIBUTION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "LEVEL DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    for level, count in levels.most_common():

        percentage = (
            count
            / total
            * 100
        )

        print(
            f"{level:<25}"
            f"{count:>7}"
            f" ({percentage:.2f}%)"
        )

    # ========================================================
    # SOURCE DISTRIBUTION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SOURCE DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    for source, count in sources.most_common():

        percentage = (
            count
            / total
            * 100
        )

        print(
            f"{source:<45}"
            f"{count:>7}"
            f" ({percentage:.2f}%)"
        )

    # ========================================================
    # PROBLEMATIC SAMPLES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PROBLEMATIC RECORD SAMPLES"
    )

    print(
        "=" * 70
    )

    if problematic_records:

        for item in problematic_records:

            print(
                f"\nType   : {item['type']}"
            )

            print(
                f"Index  : {item['index']}"
            )

            print(
                f"Question: "
                f"{item['question'][:300]}"
            )

            print(
                f"Answer  : "
                f"{item['answer'][:300]}"
            )

    else:

        print(
            "\nNo obvious problematic samples found."
        )

    # ========================================================
    # FINAL QUALITY INDICATOR
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "QUALITY SUMMARY"
    )

    print(
        "=" * 70
    )

    issues = (

        invalid_json
        + empty_questions
        + empty_answers
        + duplicate_records
        + very_short_answers
        + suspicious_records

    )

    if issues == 0:

        print(
            "\n🟢 Dataset looks very clean."
        )

    elif issues < total * 0.05:

        print(
            "\n🟢 Dataset quality looks GOOD."
        )

        print(
            "Only a small percentage "
            "of records need attention."
        )

    elif issues < total * 0.10:

        print(
            "\n🟡 Dataset needs SOME cleaning."
        )

    else:

        print(
            "\n🔴 Dataset needs significant cleaning."
        )

    # ========================================================
    # IMPORTANT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "IMPORTANT"
    )

    print(
        "=" * 70
    )

    print(
        "\nThis script ONLY analyzes the dataset."
    )

    print(
        "It does NOT delete any records."
    )

    print(
        "It does NOT modify merged_dataset.jsonl."
    )

    print(
        "\nDone!"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()