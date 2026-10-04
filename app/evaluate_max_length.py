import json
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

FILES = {
    "train": "dataset/splits/train.jsonl",
    "validation": "dataset/splits/validation.jsonl",
    "test": "dataset/splits/test.jsonl",
}

CANDIDATE_LENGTHS = [256, 384, 512, 1024]


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


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


def analyze_file(file_path):
    lengths = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            example = json.loads(line)
            lengths.append(get_token_length(example))

    return lengths


for split_name, file_path in FILES.items():

    print("\n" + "=" * 70)
    print(f"{split_name.upper()}")
    print("=" * 70)

    lengths = analyze_file(file_path)

    total = len(lengths)

    print(f"Total examples: {total}")

    for max_length in CANDIDATE_LENGTHS:

        truncated = sum(
            length > max_length
            for length in lengths
        )

        percentage = (truncated / total) * 100

        retained = total - truncated
        retained_percentage = (retained / total) * 100

        print(
            f"\nmax_length = {max_length}"
        )
        print(
            f"  Truncated : {truncated}"
        )
        print(
            f"  Truncated % : {percentage:.2f}%"
        )
        print(
            f"  Fully retained : {retained}"
        )
        print(
            f"  Retained % : {retained_percentage:.2f}%"
        )


print("\n" + "=" * 70)
print("MAX_LENGTH EVALUATION COMPLETE")
print("=" * 70)