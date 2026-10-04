import json
import statistics
from transformers import AutoTokenizer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

FILES = {
    "train": "dataset/splits/train.jsonl",
    "validation": "dataset/splits/validation.jsonl",
    "test": "dataset/splits/test.jsonl",
}


# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded.")


# --------------------------------------------------
# 3. Function to calculate token length
# --------------------------------------------------

def get_token_length(example):
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

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    encoded = tokenizer(
        formatted_text,
        add_special_tokens=False,
    )

    return len(encoded["input_ids"])


# --------------------------------------------------
# 4. Calculate statistics for one dataset split
# --------------------------------------------------

def analyze_file(file_path):
    lengths = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            example = json.loads(line)
            length = get_token_length(example)
            lengths.append(length)

    if not lengths:
        return None

    lengths_sorted = sorted(lengths)

    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "p90": lengths_sorted[int(0.90 * len(lengths)) - 1],
        "p95": lengths_sorted[int(0.95 * len(lengths)) - 1],
        "p99": lengths_sorted[int(0.99 * len(lengths)) - 1],
    }


# --------------------------------------------------
# 5. Analyze train / validation / test
# --------------------------------------------------

for split_name, file_path in FILES.items():

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} TOKEN LENGTH")
    print("=" * 60)

    stats = analyze_file(file_path)

    if stats is None:
        print("No examples found.")
        continue

    print(f"Number of examples : {stats['count']}")
    print(f"Minimum            : {stats['min']}")
    print(f"Maximum            : {stats['max']}")
    print(f"Mean               : {stats['mean']:.2f}")
    print(f"Median             : {stats['median']}")
    print(f"90th percentile    : {stats['p90']}")
    print(f"95th percentile    : {stats['p95']}")
    print(f"99th percentile    : {stats['p99']}")


print("\n" + "=" * 60)
print("TOKEN LENGTH ANALYSIS COMPLETE")
print("=" * 60)