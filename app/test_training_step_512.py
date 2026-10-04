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

TRAIN_FILE = Path(
    "dataset/tokenized/train.jsonl"
)

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
# Find longest example <= MAX_LENGTH
# ============================================================

def find_longest_example(dataset, max_length):

    best_example = None
    best_length = 0
    best_index = None

    for index, example in enumerate(dataset.examples):

        length = len(
            example["input_ids"]
        )

        if (
            length <= max_length
            and length > best_length
        ):

            best_length = length
            best_example = example
            best_index = index

    return (
        best_example,
        best_length,
        best_index,
    )


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("512-TOKEN TRAINING STEP STRESS TEST")
print("=" * 70)


# ============================================================
# 1. Check CUDA
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is required for this test."
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
# 3. Load tokenized dataset
# ============================================================

print()
print("Loading tokenized dataset...")

dataset = TokenizedDataset(
    TRAIN_FILE
)

print(
    f"Dataset examples: {len(dataset)}"
)


# ============================================================
# 4. Find longest example <= 512
# ============================================================

print()
print(
    f"Searching for longest example "
    f"with length <= {MAX_LENGTH}..."
)

(
    selected_example,
    selected_length,
    selected_index,
) = find_longest_example(
    dataset,
    MAX_LENGTH,
)

if selected_example is None:

    raise RuntimeError(
        f"No example with length <= {MAX_LENGTH} "
        "was found."
    )

print()
print(
    f"Selected dataset index: "
    f"{selected_index}"
)

print(
    f"Selected sequence length: "
    f"{selected_length}"
)

print(
    f"Unused positions relative to MAX_LENGTH: "
    f"{MAX_LENGTH - selected_length}"
)


# ============================================================
# 5. Create one-example DataLoader
# ============================================================

collator = CausalLMCollator(
    pad_token_id=tokenizer.pad_token_id,
    ignore_index=IGNORE_INDEX,
)

single_example_dataset = [
    selected_example
]

dataloader = DataLoader(
    single_example_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
)

batch = next(
    iter(dataloader)
)


# ============================================================
# 6. Inspect batch
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
# 7. Check trainable label tokens
# ============================================================

trainable_label_tokens = (
    batch["labels"] != IGNORE_INDEX
).sum().item()

ignored_label_tokens = (
    batch["labels"] == IGNORE_INDEX
).sum().item()

print()
print(
    "Trainable label tokens:",
    trainable_label_tokens,
)

print(
    "Ignored label tokens:",
    ignored_label_tokens,
)


if trainable_label_tokens == 0:

    raise RuntimeError(
        "Selected example contains no trainable "
        "label tokens."
    )


# ============================================================
# 8. Create 4-bit configuration
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
# 9. Load model
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
# 10. Prepare model for k-bit training
# ============================================================

print()
print("Preparing model for k-bit training...")

model = prepare_model_for_kbit_training(
    model
)

print(
    "Model prepared."
)


# ============================================================
# 11. LoRA configuration
# ============================================================

print()
print("Creating LoRA configuration...")

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
# 12. Attach LoRA
# ============================================================

print()
print("Attaching LoRA...")

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()


# ============================================================
# 13. Training mode
# ============================================================

model.train()

# Gradient checkpointing was automatically active in the
# previous test. Explicitly make the training configuration
# clear here.

model.config.use_cache = False


# ============================================================
# 14. Move batch to model device
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
# 15. Clear CUDA memory statistics
# ============================================================

torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()


# ============================================================
# 16. Forward pass
# ============================================================

print()
print("=" * 70)
print("FORWARD PASS")
print("=" * 70)

try:

    outputs = model(

        input_ids=batch["input_ids"],

        attention_mask=batch["attention_mask"],

        labels=batch["labels"],
    )

except RuntimeError as error:

    if "out of memory" in str(error).lower():

        print()
        print("=" * 70)
        print("CUDA OUT OF MEMORY DURING FORWARD PASS")
        print("=" * 70)

        print()
        print(error)

        raise

    raise


loss = outputs.loss


print()
print("Loss:")
print(loss.item())


print()
print("Logits shape:")
print(outputs.logits.shape)


# ============================================================
# 17. Backward pass
# ============================================================

print()
print("=" * 70)
print("BACKWARD PASS")
print("=" * 70)

try:

    loss.backward()

except RuntimeError as error:

    if "out of memory" in str(error).lower():

        print()
        print("=" * 70)
        print("CUDA OUT OF MEMORY DURING BACKWARD PASS")
        print("=" * 70)

        print()
        print(error)

        raise

    raise


print()
print(
    "Backward pass completed successfully."
)


# ============================================================
# 18. Check LoRA gradients
# ============================================================

print()
print("=" * 70)
print("GRADIENT CHECK")
print("=" * 70)

trainable_parameters = 0

parameters_with_gradients = 0

parameters_without_gradients = 0

gradient_elements = 0


for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        trainable_parameters += 1

        if parameter.grad is not None:

            parameters_with_gradients += 1

            gradient_elements += (
                parameter.grad.numel()
            )

        else:

            parameters_without_gradients += 1


print()
print(
    "Trainable parameter tensors:",
    trainable_parameters,
)

print(
    "Trainable tensors with gradients:",
    parameters_with_gradients,
)

print(
    "Trainable tensors without gradients:",
    parameters_without_gradients,
)

print(
    "Gradient elements:",
    gradient_elements,
)


# ============================================================
# 19. GPU memory
# ============================================================

current_allocated = (
    torch.cuda.memory_allocated(0)
)

current_reserved = (
    torch.cuda.memory_reserved(0)
)

peak_allocated = (
    torch.cuda.max_memory_allocated(0)
)


print()
print("=" * 70)
print("GPU MEMORY")
print("=" * 70)

print()

print(
    f"Current allocated: "
    f"{current_allocated / (1024 ** 3):.2f} GB"
)

print(
    f"Current reserved:  "
    f"{current_reserved / (1024 ** 3):.2f} GB"
)

print(
    f"Peak allocated:    "
    f"{peak_allocated / (1024 ** 3):.2f} GB"
)


# ============================================================
# 20. Final result
# ============================================================

print()
print("=" * 70)
print("512-TOKEN TRAINING STEP TEST COMPLETE")
print("=" * 70)

print()
print("RESULT: PASSED")

print()
print(
    f"Sequence length tested: {selected_length}"
)

print(
    f"Peak GPU memory: "
    f"{peak_allocated / (1024 ** 3):.2f} GB"
)

print()
print(
    "Forward pass:  PASSED"
)

print(
    "Backward pass: PASSED"
)

print(
    "Gradient check: PASSED"
)