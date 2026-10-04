import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = Path("dataset/tokenized/train.jsonl")

BATCH_SIZE = 1

MAX_LENGTH = 512

IGNORE_INDEX = -100


# ============================================================
# Dataset
# ============================================================

class TokenizedDataset(Dataset):

    def __init__(self, file_path):

        self.examples = []

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                if line.strip():

                    self.examples.append(
                        json.loads(line)
                    )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        return self.examples[index]


# ============================================================
# Collator
# ============================================================

class CausalLMCollator:

    def __init__(
        self,
        pad_token_id,
        ignore_index=-100,
    ):

        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, examples):

        max_length = max(
            len(example["input_ids"])
            for example in examples
        )

        input_ids = []
        attention_mask = []
        labels = []

        for example in examples:

            padding_length = (
                max_length
                - len(example["input_ids"])
            )

            input_ids.append(
                example["input_ids"]
                + [self.pad_token_id] * padding_length
            )

            attention_mask.append(
                example["attention_mask"]
                + [0] * padding_length
            )

            labels.append(
                example["labels"]
                + [self.ignore_index] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("SINGLE TRAINING STEP TEST")
print("=" * 70)


# ============================================================
# 1. Check GPU
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is required."
    )

print()
print("GPU:")
print(torch.cuda.get_device_name(0))


# ============================================================
# 2. Load tokenizer
# ============================================================

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")


# ============================================================
# 3. Load dataset
# ============================================================

print()
print("Loading dataset...")

dataset = TokenizedDataset(
    TRAIN_FILE
)

print(
    f"Dataset examples: {len(dataset)}"
)


# ============================================================
# 4. Create DataLoader
# ============================================================

collator = CausalLMCollator(
    pad_token_id=tokenizer.pad_token_id,
    ignore_index=IGNORE_INDEX,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
)

batch = next(iter(dataloader))


# ============================================================
# 5. Display batch shape
# ============================================================

print()
print("=" * 70)
print("BATCH")
print("=" * 70)

print()
print(
    "input_ids shape:",
    batch["input_ids"].shape,
)

print(
    "attention_mask shape:",
    batch["attention_mask"].shape,
)

print(
    "labels shape:",
    batch["labels"].shape,
)


# ============================================================
# 6. 4-bit configuration
# ============================================================

print()
print("Creating 4-bit configuration...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 7. Load model
# ============================================================

print()
print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
)

print("Model loaded.")


# ============================================================
# 8. Prepare for k-bit training
# ============================================================

print()
print("Preparing model...")

model = prepare_model_for_kbit_training(
    model
)


# ============================================================
# 9. LoRA configuration
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)


# ============================================================
# 10. Attach LoRA
# ============================================================

print()
print("Attaching LoRA...")

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()


# ============================================================
# 11. Move batch to GPU
# ============================================================

device = model.device

print()
print("Model device:")
print(device)

batch = {
    key: value.to(device)
    for key, value in batch.items()
}


# ============================================================
# 12. Clear memory before test
# ============================================================

torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()


# ============================================================
# 13. Forward pass
# ============================================================

print()
print("=" * 70)
print("FORWARD PASS")
print("=" * 70)

model.train()

outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)

loss = outputs.loss

print()
print("Loss:")
print(loss.item())

print()
print("Logits shape:")
print(outputs.logits.shape)


# ============================================================
# 14. Backward pass
# ============================================================

print()
print("=" * 70)
print("BACKWARD PASS")
print("=" * 70)

loss.backward()

print()
print("Backward pass completed successfully.")


# ============================================================
# 15. Check LoRA gradients
# ============================================================

print()
print("=" * 70)
print("GRADIENT CHECK")
print("=" * 70)

gradient_parameters = 0

for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        if parameter.grad is not None:

            gradient_parameters += 1


print()
print(
    "Trainable parameters with gradients:",
    gradient_parameters,
)


# ============================================================
# 16. GPU memory
# ============================================================

allocated = torch.cuda.memory_allocated(0)

reserved = torch.cuda.memory_reserved(0)

peak = torch.cuda.max_memory_allocated(0)


print()
print("=" * 70)
print("GPU MEMORY")
print("=" * 70)

print()
print(
    f"Current allocated: "
    f"{allocated / (1024 ** 3):.2f} GB"
)

print(
    f"Current reserved:  "
    f"{reserved / (1024 ** 3):.2f} GB"
)

print(
    f"Peak allocated:    "
    f"{peak / (1024 ** 3):.2f} GB"
)


# ============================================================
# Final
# ============================================================

print()
print("=" * 70)
print("SINGLE TRAINING STEP TEST COMPLETE")
print("=" * 70)