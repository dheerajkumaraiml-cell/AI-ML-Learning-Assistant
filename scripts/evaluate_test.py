import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

CHECKPOINT = Path(
    "checkpoints/qlora_qwen/checkpoint-3500"
)

TEST_FILE = Path(
    "dataset/tokenized/test.jsonl"
)

BATCH_SIZE = 1

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
# Data collator
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
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
):

    model.eval()

    total_loss = 0.0
    batches = 0

    for batch in dataloader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss

        total_loss += loss.item()
        batches += 1

        if batches % 100 == 0:

            print(
                f"Evaluated {batches} batches | "
                f"Current loss: {loss.item():.4f}"
            )

    if batches == 0:

        return float("nan")

    return total_loss / batches


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    device = torch.device("cuda:0")

    print()
    print("GPU:")
    print(torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Validate checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    adapter_config = (
        CHECKPOINT / "adapter_config.json"
    )

    if not adapter_config.exists():

        raise FileNotFoundError(
            f"LoRA adapter not found:\n{adapter_config}"
        )

    print()
    print(f"Checkpoint: {CHECKPOINT}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded.")

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    print()
    print("Loading test dataset...")

    test_dataset = TokenizedDataset(
        TEST_FILE
    )

    print(
        f"Test examples: {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    collator = CausalLMCollator(
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=IGNORE_INDEX,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    # --------------------------------------------------------
    # 4-bit configuration
    # --------------------------------------------------------

    print()
    print("Creating 4-bit configuration...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --------------------------------------------------------
    # Load base model
    # --------------------------------------------------------

    print()
    print("Loading base Qwen model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )

    print("Base model loaded.")

    # --------------------------------------------------------
    # Load LoRA adapter
    # --------------------------------------------------------

    print()
    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        CHECKPOINT,
        is_trainable=False,
    )

    print("LoRA adapter loaded.")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TEST EVALUATION")
    print("=" * 70)

    test_loss = evaluate(
        model,
        test_dataloader,
        device,
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Test loss: {test_loss:.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()