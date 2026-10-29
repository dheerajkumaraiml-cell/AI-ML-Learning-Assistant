
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
    """Load tokenized examples from a JSONL file."""

    def __init__(self, file_path):
        self.examples = []

        if not file_path.exists():
            raise FileNotFoundError(
                f"Test dataset not found: {file_path}"
            )

        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    example = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number} "
                        f"in {file_path}"
                    ) from exc

                required_fields = {
                    "input_ids",
                    "attention_mask",
                    "labels",
                }

                missing_fields = (
                    required_fields - example.keys()
                )

                if missing_fields:
                    raise ValueError(
                        f"Line {line_number} is missing fields: "
                        f"{sorted(missing_fields)}"
                    )

                if not (
                    len(example["input_ids"])
                    == len(example["attention_mask"])
                    == len(example["labels"])
                ):
                    raise ValueError(
                        f"Sequence lengths do not match at "
                        f"line {line_number}."
                    )

                if not example["input_ids"]:
                    raise ValueError(
                        f"Empty input sequence at line {line_number}."
                    )

                self.examples.append(example)

        if not self.examples:
            raise ValueError(
                f"Test dataset is empty: {file_path}"
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ============================================================
# Data collator
# ============================================================

class CausalLMCollator:
    """Pad sequences while excluding padding from the loss."""

    def __init__(self, pad_token_id, ignore_index=-100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, examples):
        max_length = max(
            len(example["input_ids"])
            for example in examples
        )

        input_ids = []
        attention_masks = []
        labels = []

        for example in examples:
            padding_length = (
                max_length - len(example["input_ids"])
            )

            input_ids.append(
                example["input_ids"]
                + [self.pad_token_id] * padding_length
            )

            attention_masks.append(
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
                attention_masks,
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
def evaluate(model, dataloader, device):
    """
    Calculate token-weighted causal language modeling loss.

    The model shifts labels internally for next-token prediction.
    Prompt and padding labels marked IGNORE_INDEX are excluded.
    """

    model.eval()

    total_weighted_loss = 0.0
    total_valid_tokens = 0
    total_examples = 0

    for batch_number, batch in enumerate(
        dataloader,
        start=1,
    ):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Causal LM loss uses labels shifted one position to the
        # left relative to the logits. Position zero is not a
        # next-token prediction target.
        valid_tokens = (
            batch["labels"][:, 1:] != IGNORE_INDEX
        ).sum().item()

        if valid_tokens == 0:
            continue

        batch_loss = outputs.loss.item()

        total_weighted_loss += (
            batch_loss * valid_tokens
        )

        total_valid_tokens += valid_tokens
        total_examples += batch["input_ids"].size(0)

        if batch_number % 100 == 0:
            print(
                f"Evaluated {batch_number} batches | "
                f"Last batch loss: {batch_loss:.4f}"
            )

    if total_valid_tokens == 0:
        raise ValueError(
            "No valid target tokens were found in the test dataset."
        )

    average_loss = (
        total_weighted_loss / total_valid_tokens
    )

    return {
        "test_loss": average_loss,
        "valid_target_tokens": total_valid_tokens,
        "evaluated_examples": total_examples,
    }


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
            "CUDA GPU is required for this evaluation."
        )

    device = torch.device("cuda:0")

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not CHECKPOINT.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {CHECKPOINT}"
        )

    adapter_config = CHECKPOINT / "adapter_config.json"

    if not adapter_config.is_file():
        raise FileNotFoundError(
            f"LoRA adapter configuration not found: "
            f"{adapter_config}"
        )

    if not TEST_FILE.is_file():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    print(f"\nBase model: {MODEL_NAME}")
    print(f"LoRA checkpoint: {CHECKPOINT}")
    print(f"Test dataset: {TEST_FILE}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither a pad token nor an EOS token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded.")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    test_dataset = TokenizedDataset(TEST_FILE)

    print(f"Test examples: {len(test_dataset)}")

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
        num_workers=0,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # 4-bit quantization
    # --------------------------------------------------------

    print("\nCreating 4-bit quantization configuration...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --------------------------------------------------------
    # Load base model
    # --------------------------------------------------------

    print("\nLoading base Qwen model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map={"": 0},
    )

    print("Base model loaded.")

    # --------------------------------------------------------
    # Load trained LoRA adapter
    # --------------------------------------------------------

    print("\nLoading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        CHECKPOINT,
        is_trainable=False,
    )

    model.eval()

    print("LoRA adapter loaded.")
    print("Model is in evaluation mode.")

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING TEST EVALUATION")
    print("=" * 70)

    results = evaluate(
        model=model,
        dataloader=test_dataloader,
        device=device,
    )

    # --------------------------------------------------------
    # Report results
    # --------------------------------------------------------

    test_loss = results["test_loss"]

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(f"Test loss: {test_loss:.4f}")
    print(
        f"Evaluated examples: "
        f"{results['evaluated_examples']}"
    )
    print(
        f"Valid target tokens: "
        f"{results['valid_target_tokens']}"
    )

    # Perplexity is exp(loss). Very large losses can overflow,
    # so report it only when the exponent is within a safe range.
    if test_loss < 50:
        perplexity = torch.exp(
            torch.tensor(test_loss, dtype=torch.float64)
        ).item()

        print(f"Perplexity: {perplexity:.4f}")
    else:
        print(
            "Perplexity: not calculated because the loss is "
            "too large for a useful direct exponentiation."
        )

    print("=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
