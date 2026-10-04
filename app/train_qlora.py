import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    PeftModel,
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

VALIDATION_FILE = Path(
    "dataset/tokenized/validation.jsonl"
)

OUTPUT_DIR = Path(
    "checkpoints/qlora_qwen"
)

BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

NUM_EPOCHS = 2

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.03

MAX_LENGTH = 512

IGNORE_INDEX = -100

LOG_EVERY_STEPS = 25

EVAL_EVERY_STEPS = 500

SAVE_EVERY_STEPS = 100

SEED = 42


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train Qwen with QLoRA."
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory to resume from, "
            "for example: "
            "checkpoints/qlora_qwen/checkpoint-500"
        ),
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


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
                + [
                    self.pad_token_id
                ] * padding_length
            )

            attention_mask.append(
                example["attention_mask"]
                + [0] * padding_length
            )

            labels.append(
                example["labels"]
                + [
                    self.ignore_index
                ] * padding_length
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

    model.train()

    if batches == 0:

        return float("nan")

    return total_loss / batches


# ============================================================
# Checkpoint saving
# ============================================================

def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    output_dir,
    global_step,
    epoch,
):

    checkpoint_dir = (
        output_dir
        / f"checkpoint-{global_step}"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Saving checkpoint: "
        f"{checkpoint_dir}"
    )

    # Save LoRA adapter.
    model.save_pretrained(
        checkpoint_dir
    )

    # Save tokenizer and chat template.
    tokenizer.save_pretrained(
        checkpoint_dir
    )

    # Save optimizer, scheduler, and progress state.
    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_dir
        / "training_state.pt",
    )

    print(
        "Checkpoint saved."
    )


# ============================================================
# Load training state
# ============================================================

def load_training_state(
    checkpoint_dir,
    optimizer,
    scheduler,
):

    training_state_file = (
        checkpoint_dir
        / "training_state.pt"
    )

    if not training_state_file.exists():

        raise FileNotFoundError(
            f"Training state not found:\n"
            f"{training_state_file}"
        )

    print()
    print(
        "Loading training state..."
    )

    state = torch.load(
        training_state_file,
        map_location="cpu",
    )

    optimizer.load_state_dict(
        state["optimizer"]
    )

    scheduler.load_state_dict(
        state["scheduler"]
    )

    global_step = state[
        "global_step"
    ]

    saved_epoch = state[
        "epoch"
    ]

    print(
        f"Resumed global step: "
        f"{global_step}"
    )

    print(
        f"Checkpoint epoch value: "
        f"{saved_epoch}"
    )

    return global_step, saved_epoch


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    resume_checkpoint = (
        args.resume
    )

    # --------------------------------------------------------
    # Validate resume path
    # --------------------------------------------------------

    if resume_checkpoint is not None:

        if not resume_checkpoint.exists():

            raise FileNotFoundError(
                f"Checkpoint directory does not exist:\n"
                f"{resume_checkpoint}"
            )

        adapter_file = (
            resume_checkpoint
            / "adapter_config.json"
        )

        if not adapter_file.exists():

            raise FileNotFoundError(
                f"LoRA adapter not found:\n"
                f"{adapter_file}"
            )

        print()
        print(
            "=" * 70
        )

        print(
            "RESUME MODE"
        )

        print(
            "=" * 70
        )

        print()
        print(
            f"Checkpoint: "
            f"{resume_checkpoint}"
        )

    else:

        print(
            "=" * 70
        )

        print(
            "NEW QLORA TRAINING RUN"
        )

        print(
            "=" * 70
        )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    device = torch.device(
        "cuda:0"
    )

    print()
    print("GPU:")

    print(
        torch.cuda.get_device_name(0)
    )

    total_gpu_memory = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU memory: "
        f"{total_gpu_memory:.2f} GB"
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print()
    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        "Tokenizer loaded."
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading datasets..."
    )

    train_dataset = TokenizedDataset(
        TRAIN_FILE
    )

    validation_dataset = TokenizedDataset(
        VALIDATION_FILE
    )

    print(
        f"Training examples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    collator = CausalLMCollator(
        pad_token_id=(
            tokenizer.pad_token_id
        ),
        ignore_index=IGNORE_INDEX,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    # --------------------------------------------------------
    # Training calculations
    # --------------------------------------------------------

    updates_per_epoch = math.ceil(
        len(train_dataloader)
        / GRADIENT_ACCUMULATION_STEPS
    )

    total_training_steps = (
        updates_per_epoch
        * NUM_EPOCHS
    )

    warmup_steps = int(
        total_training_steps
        * WARMUP_RATIO
    )

    effective_batch_size = (
        BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    print()
    print(
        "=" * 70
    )

    print(
        "TRAINING CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print()
    print(
        f"Physical batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Effective batch size: "
        f"{effective_batch_size}"
    )

    print(
        f"Max sequence length: "
        f"{MAX_LENGTH}"
    )

    print(
        f"Epochs: "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Warmup steps: "
        f"{warmup_steps}"
    )

    print(
        f"Updates per epoch: "
        f"{updates_per_epoch}"
    )

    print(
        f"Total training steps: "
        f"{total_training_steps}"
    )

    # --------------------------------------------------------
    # 4-bit configuration
    # --------------------------------------------------------

    print()
    print(
        "Creating 4-bit configuration..."
    )

    quantization_config = (
        BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            bnb_4bit_compute_dtype=(
                torch.float16
            ),

            bnb_4bit_use_double_quant=True,
        )
    )

    # --------------------------------------------------------
    # Load base model
    # --------------------------------------------------------

    print()
    print(
        "Loading base model..."
    )

    model = (
        AutoModelForCausalLM.from_pretrained(

            MODEL_NAME,

            quantization_config=(
                quantization_config
            ),

            device_map="auto",
        )
    )

    print(
        "Base model loaded."
    )

    # --------------------------------------------------------
    # Prepare k-bit training
    # --------------------------------------------------------

    print()
    print(
        "Preparing model for k-bit training..."
    )

    model = (
        prepare_model_for_kbit_training(
            model
        )
    )

    # --------------------------------------------------------
    # Gradient checkpointing
    # --------------------------------------------------------

    print()
    print(
        "Enabling gradient checkpointing..."
    )

    model.gradient_checkpointing_enable()

    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA configuration
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # New LoRA or resume LoRA
    # --------------------------------------------------------

    if resume_checkpoint is None:

        print()
        print(
            "Attaching new LoRA adapter..."
        )

        model = get_peft_model(
            model,
            lora_config,
        )

    else:

        print()
        print(
            "Loading LoRA adapter from checkpoint..."
        )

        model = PeftModel.from_pretrained(

            model,

            resume_checkpoint,

            is_trainable=True,
        )

    model.print_trainable_parameters()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    trainable_parameters = [

        parameter

        for parameter in model.parameters()

        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(

        trainable_parameters,

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        get_linear_schedule_with_warmup(

            optimizer,

            num_warmup_steps=warmup_steps,

            num_training_steps=(
                total_training_steps
            ),
        )
    )

    # --------------------------------------------------------
    # Restore state if resuming
    # --------------------------------------------------------

    global_step = 0

    start_epoch = 0

    if resume_checkpoint is not None:

        (
            global_step,
            saved_epoch,
        ) = load_training_state(

            resume_checkpoint,

            optimizer,

            scheduler,
        )

        # The checkpoint was saved in the middle
        # of epoch 1. We intentionally restart
        # the DataLoader from the beginning of
        # the epoch rather than pretending to have
        # exact sampler/batch-position state.

        start_epoch = 0

        print()
        print(
            "Resume strategy:"
        )

        print(
            "Restoring model/optimizer/scheduler "
            "state."
        )

        print(
            "Restarting the current epoch's "
            "DataLoader from its beginning."
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    print()
    print(
        "=" * 70
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "=" * 70
    )

    print()

    for epoch in range(
        start_epoch,
        NUM_EPOCHS,
    ):

        model.train()

        epoch_loss = 0.0

        micro_steps = 0

        print()
        print(
            f"Epoch "
            f"{epoch + 1}/{NUM_EPOCHS}"
        )

        for step, batch in enumerate(
            train_dataloader
        ):

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            outputs = model(

                input_ids=batch[
                    "input_ids"
                ],

                attention_mask=batch[
                    "attention_mask"
                ],

                labels=batch[
                    "labels"
                ],
            )

            loss = outputs.loss

            epoch_loss += loss.item()

            micro_steps += 1

            # ------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------

            loss_for_backward = (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            loss_for_backward.backward()

            # ------------------------------------------------
            # Optimizer update
            # ------------------------------------------------

            accumulation_complete = (

                micro_steps
                % GRADIENT_ACCUMULATION_STEPS
                == 0
            )

            last_batch = (
                step + 1
                == len(train_dataloader)
            )

            if (
                accumulation_complete
                or last_batch
            ):

                optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

                # ------------------------------------------------
                # Logging
                # ------------------------------------------------

                if (
                    global_step
                    % LOG_EVERY_STEPS
                    == 0
                ):

                    current_lr = (
                        scheduler
                        .get_last_lr()[0]
                    )

                    print(
                        f"step={global_step} "
                        f"epoch={epoch + 1} "
                        f"loss={loss.item():.4f} "
                        f"lr={current_lr:.8f}"
                    )

                # ------------------------------------------------
                # Validation
                # ------------------------------------------------

                if (
                    global_step
                    % EVAL_EVERY_STEPS
                    == 0
                ):

                    print()
                    print(
                        f"Running validation "
                        f"at step "
                        f"{global_step}..."
                    )

                    validation_loss = evaluate(

                        model,

                        validation_dataloader,

                        device,
                    )

                    print(
                        f"Validation loss: "
                        f"{validation_loss:.4f}"
                    )

                # ------------------------------------------------
                # Checkpoint
                # ------------------------------------------------

                if (
                    global_step
                    % SAVE_EVERY_STEPS
                    == 0
                ):

                    save_checkpoint(

                        model=model,

                        tokenizer=tokenizer,

                        optimizer=optimizer,

                        scheduler=scheduler,

                        output_dir=OUTPUT_DIR,

                        global_step=global_step,

                        epoch=epoch + 1,
                    )

        # --------------------------------------------------------
        # End epoch
        # --------------------------------------------------------

        average_epoch_loss = (

            epoch_loss
            / micro_steps
        )

        print()
        print(
            f"Epoch {epoch + 1} complete."
        )

        print(
            f"Average training loss: "
            f"{average_epoch_loss:.4f}"
        )

        print()
        print(
            "Running end-of-epoch validation..."
        )

        validation_loss = evaluate(

            model,

            validation_dataloader,

            device,
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL SAVE"
    )

    print(
        "=" * 70
    )

    save_checkpoint(

        model=model,

        tokenizer=tokenizer,

        optimizer=optimizer,

        scheduler=scheduler,

        output_dir=OUTPUT_DIR,

        global_step=global_step,

        epoch=NUM_EPOCHS,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print()
    print(
        f"Final optimizer step: "
        f"{global_step}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()