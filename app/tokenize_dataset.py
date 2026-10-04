import json
from pathlib import Path

from transformers import AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

INPUT_DIR = Path("dataset/splits")
OUTPUT_DIR = Path("dataset/tokenized")

MAX_LENGTH = 512
IGNORE_INDEX = -100

SPLITS = ["train", "validation", "test"]


# ============================================================
# Load tokenizer
# ============================================================

print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Model: {MODEL_NAME}")
print(f"Max length: {MAX_LENGTH}")


# ============================================================
# Tokenize one example
# ============================================================

def tokenize_example(example):
    """
    Convert one Q/A example into model-ready training data.

    Returns:
        input_ids
        attention_mask
        labels
        was_truncated
    """

    messages = [
        {
            "role": "user",
            "content": example["question"],
        },
        {
            "role": "assistant",
            "content": example["answer"],
        },
    ]

    # --------------------------------------------------------
    # Complete conversation
    # --------------------------------------------------------

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # --------------------------------------------------------
    # Tokenize complete conversation
    # --------------------------------------------------------

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
        return_attention_mask=True,
    )

    input_ids = full_ids["input_ids"]
    attention_mask = full_ids["attention_mask"]

    # --------------------------------------------------------
    # Find assistant start
    # --------------------------------------------------------

    prompt_messages = [
        {
            "role": "user",
            "content": example["question"],
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    assistant_start = len(prompt_ids)

    # --------------------------------------------------------
    # Detect truncation
    # --------------------------------------------------------

    was_truncated = len(full_ids["input_ids"]) >= MAX_LENGTH

    # --------------------------------------------------------
    # Create labels
    # --------------------------------------------------------

    labels = []

    for position, token_id in enumerate(input_ids):

        if position < assistant_start:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "was_truncated": was_truncated,
    }


# ============================================================
# Process one dataset split
# ============================================================

def process_split(split_name):
    input_file = INPUT_DIR / f"{split_name}.jsonl"
    output_file = OUTPUT_DIR / f"{split_name}.jsonl"

    print()
    print("=" * 70)
    print(f"PROCESSING: {split_name.upper()}")
    print("=" * 70)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    truncated = 0
    retained = 0
    errors = 0

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):

        for line_number, line in enumerate(infile, start=1):

            if not line.strip():
                continue

            try:
                example = json.loads(line)

                tokenized = tokenize_example(example)

                if tokenized["was_truncated"]:
                    truncated += 1
                else:
                    retained += 1

                # Remove internal inspection field
                tokenized.pop("was_truncated")

                outfile.write(
                    json.dumps(
                        tokenized,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                total += 1

            except Exception as e:
                errors += 1

                print()
                print(
                    f"ERROR at line {line_number}: {e}"
                )

    print()
    print(f"Total processed : {total}")
    print(f"Fully retained  : {retained}")
    print(f"Truncated       : {truncated}")
    print(f"Errors          : {errors}")
    print(f"Output          : {output_file}")

    return total, truncated, errors


# ============================================================
# Process all splits
# ============================================================

print()
print("=" * 70)
print("STARTING DATASET TOKENIZATION")
print("=" * 70)

results = {}

for split in SPLITS:
    results[split] = process_split(split)


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("TOKENIZATION COMPLETE")
print("=" * 70)

for split, (total, truncated, errors) in results.items():

    if total > 0:
        truncation_percentage = (
            truncated / total
        ) * 100
    else:
        truncation_percentage = 0

    print()
    print(split.upper())
    print(f"  Examples     : {total}")
    print(f"  Truncated    : {truncated}")
    print(f"  Truncated %  : {truncation_percentage:.2f}%")
    print(f"  Errors       : {errors}")

print()
print(f"Tokenized files saved to: {OUTPUT_DIR}")