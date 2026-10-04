import json
from transformers import AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "dataset/splits/train.jsonl"
MAX_LENGTH = 512


# ============================================================
# 1. Load tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded.")


# ============================================================
# 2. Read the first training example
# ============================================================

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    example = json.loads(f.readline())


# ============================================================
# 3. Convert Q/A into Qwen chat format
# ============================================================

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


# ============================================================
# 4. Apply Qwen chat template
# ============================================================

formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)


# ============================================================
# 5. Tokenize
# ============================================================

encoded = tokenizer(
    formatted_text,
    add_special_tokens=False,
    truncation=True,
    max_length=MAX_LENGTH,
    return_attention_mask=True,
)


# ============================================================
# 6. Create labels
# ============================================================

# For this inspection step, labels are initially
# identical to input_ids.
#
# Later we will change this so that the model's loss
# focuses on the assistant's response.

labels = encoded["input_ids"].copy()


# ============================================================
# 7. Display results
# ============================================================

print()
print("=" * 70)
print("TRAINING EXAMPLE INSPECTION")
print("=" * 70)


print()
print("Original question:")
print(example["question"])


print()
print("Original answer:")
print(example["answer"])


print()
print("Formatted chat:")
print(formatted_text)


print()
print("Input IDs:")
print(encoded["input_ids"])


print()
print("Attention mask:")
print(encoded["attention_mask"])


print()
print("Labels:")
print(labels)


print()
print("Number of tokens:")
print(len(encoded["input_ids"]))


print()
print("Number of labels:")
print(len(labels))


print()
print("Input IDs == Labels:")
print(encoded["input_ids"] == labels)


print()
print("Tokens:")
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

for i, token in enumerate(tokens):
    print(f"{i:4d} | {encoded['input_ids'][i]:6d} | {repr(token)}")


print()
print("Decoded input:")
print(
    tokenizer.decode(
        encoded["input_ids"],
        skip_special_tokens=False,
    )
)


print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)