import json
import re
from collections import Counter
from pathlib import Path


INPUT_FILE = Path(
    "dataset/additional/coding_debugging_cleaned.jsonl"
)


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_language(text):

    text = normalize_text(text)

    if "python" in text or "def " in text or "import pandas" in text:
        return "Python"

    if "#include" in text or "std::" in text:
        return "C/C++"

    if "public static void main" in text:
        return "Java"

    if "javascript" in text or "console.log" in text:
        return "JavaScript"

    if "php" in text or "<?php" in text:
        return "PHP"

    if "sql" in text or "select * from" in text:
        return "SQL"

    return "Other"


def analyze():

    total = 0

    languages = Counter()

    answer_lengths = []

    code_examples = 0

    short_answers = 0

    long_answers = 0

    questions = set()

    duplicates = 0

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

            question = data.get(
                "question",
                ""
            )

            answer = data.get(
                "answer",
                ""
            )

            question = str(question).strip()
            answer = str(answer).strip()

            # --------------------------------------------
            # Duplicate questions
            # --------------------------------------------

            q_key = normalize_text(question)

            if q_key in questions:
                duplicates += 1
            else:
                questions.add(q_key)

            # --------------------------------------------
            # Language
            # --------------------------------------------

            combined = question + " " + answer

            language = detect_language(
                combined
            )

            languages[language] += 1

            # --------------------------------------------
            # Answer length
            # --------------------------------------------

            length = len(answer)

            answer_lengths.append(length)

            if length < 100:
                short_answers += 1

            if length > 3000:
                long_answers += 1

            # --------------------------------------------
            # Code detection
            # --------------------------------------------

            code_patterns = [
                "```",
                "def ",
                "class ",
                "import ",
                "#include",
                "public static",
                "function ",
                "console.log",
                "SELECT ",
                "SELECT *",
                "<?php"
            ]

            if any(
                pattern.lower() in answer.lower()
                for pattern in code_patterns
            ):
                code_examples += 1

    # --------------------------------------------
    # Report
    # --------------------------------------------

    print("\n" + "=" * 60)
    print("CODING DATASET ANALYSIS")
    print("=" * 60)

    print(f"\nTotal examples       : {total}")
    print(f"Duplicate questions  : {duplicates}")
    print(f"Code examples        : {code_examples}")
    print(f"Short answers <100   : {short_answers}")
    print(f"Long answers >3000   : {long_answers}")

    if answer_lengths:

        average = (
            sum(answer_lengths)
            / len(answer_lengths)
        )

        print(
            f"Average answer size  : "
            f"{average:.0f} characters"
        )

    print("\n" + "-" * 60)
    print("LANGUAGE DISTRIBUTION")
    print("-" * 60)

    for language, count in languages.most_common():

        percentage = (
            count / total
        ) * 100

        print(
            f"{language:<20}"
            f"{count:>7}"
            f" ({percentage:.2f}%)"
        )


if __name__ == "__main__":
    analyze()