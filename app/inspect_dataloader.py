import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = Path("dataset/tokenized/train.jsonl")

BATCH_SIZE = 4


# ============================================================
# Dataset
# ============================================================

class TokenizedDataset(Dataset):
    """
    PyTorch Dataset for our tokenized training examples.

    Each item contains:
        input_ids
        attention_mask
        labels
    """

    def __init__(self, file_path):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.file_path}"
            )

        self.examples = []

        with open(
            self.file_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(f, start=1):

                if not line.strip():
                    continue

                try:
                    example = json.loads(line)

                    self.examples.append(example)

                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}: {e}"
                    )

    def __len__(self):
        """
        Return the number of examples.
        """

        return len(self.examples)

    def __getitem__(self, index):
        """
        Return one tokenized example.
        """

        example = self.examples[index]

        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["labels"],
        }


# ============================================================
# Data Collator
# ============================================================

class CausalLMCollator:
    """
    Convert variable-length examples into a padded batch.

    Padding is added only when necessary.
    """

    def __init__(self, pad_token_id, ignore_index=-100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, examples):

        # ----------------------------------------------------
        # Find the longest sequence in this batch
        # ----------------------------------------------------

        max_length = max(
            len(example["input_ids"])
            for example in examples
        )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        # ----------------------------------------------------
        # Pad every example to the batch maximum
        # ----------------------------------------------------

        for example in examples:

            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            labels = example["labels"]

            padding_length = (
                max_length - len(input_ids)
            )

            # Input IDs use the tokenizer's PAD token
            padded_input_ids = (
                input_ids
                + [self.pad_token_id] * padding_length
            )

            # Attention mask:
            # 1 = real token
            # 0 = padding
            padded_attention_mask = (
                attention_mask
                + [0] * padding_length
            )

            # Labels:
            # -100 = ignore during loss calculation
            padded_labels = (
                labels
                + [self.ignore_index] * padding_length
            )

            batch_input_ids.append(
                padded_input_ids
            )

            batch_attention_mask.append(
                padded_attention_mask
            )

            batch_labels.append(
                padded_labels
            )

        # ----------------------------------------------------
        # Convert Python lists → PyTorch tensors
        # ----------------------------------------------------

        return {
            "input_ids": torch.tensor(
                batch_input_ids,
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                batch_attention_mask,
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                batch_labels,
                dtype=torch.long,
            ),
        }


# ============================================================
# Load Dataset
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

dataset = TokenizedDataset(TRAIN_FILE)

print()
print(f"Dataset file : {TRAIN_FILE}")
print(f"Examples     : {len(dataset)}")


# ============================================================
# Inspect one example
# ============================================================

print()
print("=" * 70)
print("SINGLE EXAMPLE")
print("=" * 70)

example = dataset[0]

print()
print("Keys:")
print(example.keys())

print()
print("input_ids length:")
print(len(example["input_ids"]))

print()
print("attention_mask length:")
print(len(example["attention_mask"]))

print()
print("labels length:")
print(len(example["labels"]))


# ============================================================
# Create DataLoader
# ============================================================

print()
print("=" * 70)
print("CREATING DATALOADER")
print("=" * 70)

# Qwen tokenizer uses a PAD token that we should use for
# batch padding.
#
# If no PAD token exists, use EOS as a fallback.

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

pad_token_id = tokenizer.pad_token_id

print()
print(f"PAD token     : {tokenizer.pad_token}")
print(f"PAD token ID  : {pad_token_id}")


collator = CausalLMCollator(
    pad_token_id=pad_token_id,
    ignore_index=-100,
)


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
)


# ============================================================
# Get one batch
# ============================================================

print()
print("=" * 70)
print("FIRST BATCH")
print("=" * 70)

batch = next(iter(dataloader))


# ============================================================
# Inspect batch
# ============================================================

print()
print("Batch keys:")
print(batch.keys())

print()
print("input_ids shape:")
print(batch["input_ids"].shape)

print()
print("attention_mask shape:")
print(batch["attention_mask"].shape)

print()
print("labels shape:")
print(batch["labels"].shape)


# ============================================================
# Inspect tensor types
# ============================================================

print()
print("=" * 70)
print("TENSOR INFORMATION")
print("=" * 70)

print()
print("input_ids:")
print(f"  dtype : {batch['input_ids'].dtype}")
print(f"  shape : {batch['input_ids'].shape}")

print()
print("attention_mask:")
print(f"  dtype : {batch['attention_mask'].dtype}")
print(f"  shape : {batch['attention_mask'].shape}")

print()
print("labels:")
print(f"  dtype : {batch['labels'].dtype}")
print(f"  shape : {batch['labels'].shape}")


# ============================================================
# Check padding
# ============================================================

print()
print("=" * 70)
print("PADDING INSPECTION")
print("=" * 70)

print()

for i in range(BATCH_SIZE):

    attention_mask = batch["attention_mask"][i]

    real_tokens = int(
        attention_mask.sum().item()
    )

    total_tokens = len(attention_mask)

    padding_tokens = (
        total_tokens - real_tokens
    )

    print(
        f"Example {i}: "
        f"real tokens = {real_tokens}, "
        f"padding = {padding_tokens}, "
        f"total = {total_tokens}"
    )


# ============================================================
# Check ignored labels
# ============================================================

print()
print("=" * 70)
print("LABEL MASKING INSPECTION")
print("=" * 70)

for i in range(BATCH_SIZE):

    labels = batch["labels"][i]

    ignored = int(
        (labels == -100).sum().item()
    )

    trainable = int(
        (labels != -100).sum().item()
    )

    print(
        f"Example {i}: "
        f"ignored labels = {ignored}, "
        f"trainable labels = {trainable}"
    )


# ============================================================
# Display first batch
# ============================================================

print()
print("=" * 70)
print("FIRST EXAMPLE IN BATCH")
print("=" * 70)

print()
print("input_ids:")
print(batch["input_ids"][0])

print()
print("attention_mask:")
print(batch["attention_mask"][0])

print()
print("labels:")
print(batch["labels"][0])


# ============================================================
# Final validation
# ============================================================

print()
print("=" * 70)
print("VALIDATION CHECKS")
print("=" * 70)

batch_size = batch["input_ids"].shape[0]
sequence_length = batch["input_ids"].shape[1]

print()
print(f"Expected batch size : {BATCH_SIZE}")
print(f"Actual batch size   : {batch_size}")

print()
print(f"Batch sequence length: {sequence_length}")

print()
print(
    "input_ids == attention_mask shape:",
    batch["input_ids"].shape
    == batch["attention_mask"].shape,
)

print(
    "input_ids == labels shape:",
    batch["input_ids"].shape
    == batch["labels"].shape,
)

print()
print("=" * 70)
print("DATALOADER INSPECTION COMPLETE")
print("=" * 70)