import json
from transformers import AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "dataset/splits/train.jsonl"
MAX_LENGTH = 512

IGNORE_INDEX = -100


# ============================================================
# 1. Load tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded.")


# ============================================================
# 2. Load one training example
# ============================================================

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    example = json.loads(f.readline())


# ============================================================
# 3. Create chat messages
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
# 4. Create the complete conversation
# ============================================================

full_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)


# ============================================================
# 5. Tokenize the complete conversation
# ============================================================

encoded = tokenizer(
    full_text,
    add_special_tokens=False,
    truncation=True,
    max_length=MAX_LENGTH,
    return_attention_mask=True,
)

input_ids = encoded["input_ids"]


# ============================================================
# 6. Find where the assistant response starts
# ============================================================

# We create only the conversation BEFORE the assistant answer,
# but ask Qwen to add the assistant generation prompt.
#
# This gives us:
#
# system + user + assistant-start
#
# Everything before this point should be ignored by the loss.

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


# ============================================================
# 7. Create assistant-only labels
# ============================================================

labels = []

for position, token_id in enumerate(input_ids):

    if position < assistant_start:
        labels.append(IGNORE_INDEX)
    else:
        labels.append(token_id)


# ============================================================
# 8. Display basic information
# ============================================================

print()
print("=" * 70)
print("ASSISTANT-ONLY MASKING INSPECTION")
print("=" * 70)

print()
print("Question:")
print(example["question"])

print()
print("Answer:")
print(example["answer"])

print()
print("Total tokens:")
print(len(input_ids))

print()
print("Assistant starts at token position:")
print(assistant_start)

print()
print("Trainable answer tokens:")
print(sum(1 for label in labels if label != IGNORE_INDEX))

print()
print("Ignored tokens:")
print(sum(1 for label in labels if label == IGNORE_INDEX))


# ============================================================
# 9. Show token-by-token masking
# ============================================================

tokens = tokenizer.convert_ids_to_tokens(input_ids)

print()
print("=" * 70)
print("TOKEN-BY-TOKEN MASKING")
print("=" * 70)

print()
print(f"{'POS':>5} | {'TOKEN ID':>10} | {'TOKEN':<30} | LABEL")
print("-" * 70)

for position, (token_id, token, label) in enumerate(
    zip(input_ids, tokens, labels)
):

    if label == IGNORE_INDEX:
        label_display = "IGNORED (-100)"
    else:
        label_display = str(label)

    print(
        f"{position:5d} | "
        f"{token_id:10d} | "
        f"{repr(token):<30} | "
        f"{label_display}"
    )


# ============================================================
# 10. Show decoded input
# ============================================================

print()
print("=" * 70)
print("DECODED INPUT")
print("=" * 70)

print(
    tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
    )
)


# ============================================================
# 11. Show what the model actually learns from
# ============================================================

trainable_ids = [
    token_id
    for token_id, label in zip(input_ids, labels)
    if label != IGNORE_INDEX
]

print()
print("=" * 70)
print("DECODED TRAINABLE PART")
print("=" * 70)

print(
    tokenizer.decode(
        trainable_ids,
        skip_special_tokens=False,
    )
)


print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)