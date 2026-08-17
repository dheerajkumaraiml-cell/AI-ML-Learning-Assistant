import json
import re
from pathlib import Path

# --------------------------------------------------
# File paths
# --------------------------------------------------

INPUT_FILE = Path(
    "dataset/raw/concise_questions_with_answers.jsonl"
)

OUTPUT_FILE = Path(
    "dataset/cleaned_dataset.jsonl"
)


# --------------------------------------------------
# Cleaning function
# --------------------------------------------------

def clean_text(text):
    """Clean unnecessary spaces from text."""
    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


# --------------------------------------------------
# Main cleaning process
# --------------------------------------------------

def clean_dataset():

    cleaned_data = []

    seen_questions = set()

    total_rows = 0
    valid_rows = 0
    duplicate_rows = 0
    invalid_rows = 0
    short_answer_rows = 0

    # Create output directory if it doesn't exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Read JSONL
    # --------------------------------------------------

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            total_rows += 1

            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # --------------------------------------------------
            # Check valid JSON
            # --------------------------------------------------

            try:
                item = json.loads(line)

            except json.JSONDecodeError:
                invalid_rows += 1
                print(f"Invalid JSON at line {line_number}")
                continue

            # --------------------------------------------------
            # Extract fields
            # --------------------------------------------------

            question = clean_text(item.get("question"))
            answer = clean_text(item.get("answer"))
            topic = clean_text(item.get("topic"))
            level = clean_text(item.get("level"))

            # --------------------------------------------------
            # Check required fields
            # --------------------------------------------------

            if not question or not answer:
                invalid_rows += 1
                continue

            # --------------------------------------------------
            # Remove very short answers
            # --------------------------------------------------

            if len(answer) < 30:
                short_answer_rows += 1
                continue

            # --------------------------------------------------
            # Normalize question for duplicate detection
            # --------------------------------------------------

            normalized_question = question.lower()

            normalized_question = re.sub(
                r"[^\w\s]",
                "",
                normalized_question
            )

            normalized_question = re.sub(
                r"\s+",
                " ",
                normalized_question
            ).strip()

            # --------------------------------------------------
            # Remove duplicate questions
            # --------------------------------------------------

            if normalized_question in seen_questions:

                duplicate_rows += 1
                continue

            seen_questions.add(normalized_question)

            # --------------------------------------------------
            # Create cleaned record
            # --------------------------------------------------

            cleaned_item = {
                "question": question,
                "answer": answer,
                "topic": topic,
                "level": level
            }

            cleaned_data.append(cleaned_item)

            valid_rows += 1

    # --------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for item in cleaned_data:

            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )

    # --------------------------------------------------
    # Print report
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("DATASET CLEANING COMPLETE")
    print("=" * 50)

    print(f"Total rows       : {total_rows}")
    print(f"Valid rows       : {valid_rows}")
    print(f"Duplicates       : {duplicate_rows}")
    print(f"Invalid rows     : {invalid_rows}")
    print(f"Short answers    : {short_answer_rows}")
    print(f"Final dataset    : {len(cleaned_data)}")

    print("\nOutput file:")
    print(OUTPUT_FILE)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    clean_dataset()