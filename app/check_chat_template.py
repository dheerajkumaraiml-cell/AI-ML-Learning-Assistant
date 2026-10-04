import json
from transformers import AutoTokenizer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "dataset/splits/train.jsonl"

NUM_EXAMPLES = 5


# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded.")


# --------------------------------------------------
# 3. Load 5 examples
# --------------------------------------------------

examples = []

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for _ in range(NUM_EXAMPLES):
        line = f.readline()

        if not line:
            break

        examples.append(json.loads(line))


print(f"\nLoaded {len(examples)} examples.")


# --------------------------------------------------
# 4. Process each example
# --------------------------------------------------

for index, example in enumerate(examples, start=1):

    print("\n" + "=" * 70)
    print(f"EXAMPLE {index}")
    print("=" * 70)

    # ----------------------------------------------
    # Original data
    # ----------------------------------------------

    print("\nQuestion:")
    print(example["question"])

    print("\nAnswer:")
    print(example["answer"])


    # ----------------------------------------------
    # Convert Q/A → chat messages
    # ----------------------------------------------

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


    # ----------------------------------------------
    # Apply Qwen chat template
    # ----------------------------------------------

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


    print("\nFormatted text:")
    print(formatted_text)


    # ----------------------------------------------
    # Tokenize
    # ----------------------------------------------

    encoded = tokenizer(
        formatted_text,
        add_special_tokens=False,
        return_attention_mask=True,
    )


    # ----------------------------------------------
    # Extract token IDs and attention mask
    # ----------------------------------------------

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]


    # ----------------------------------------------
    # Display token information
    # ----------------------------------------------

    print("\nInput IDs:")
    print(input_ids)

    print("\nNumber of tokens:")
    print(len(input_ids))

    print("\nAttention mask:")
    print(attention_mask)


    # ----------------------------------------------
    # Convert IDs back to tokens
    # ----------------------------------------------

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("\nTokens:")
    print(tokens)


    # ----------------------------------------------
    # Decode token IDs
    # ----------------------------------------------

    decoded_text = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
    )

    print("\nDecoded text:")
    print(decoded_text)


print("\n" + "=" * 70)
print("TOKENIZATION INSPECTION COMPLETE")
print("=" * 70)