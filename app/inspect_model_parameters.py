import torch

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


# ============================================================
# 1. Quantization configuration
# ============================================================

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 2. Load tokenizer
# ============================================================

print("=" * 70)
print("MODEL PARAMETER INSPECTION")
print("=" * 70)

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 3. Load 4-bit model
# ============================================================

print()
print("Loading 4-bit model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
)

print("Model loaded.")


# ============================================================
# 4. Count parameters BEFORE LoRA
# ============================================================

print()
print("=" * 70)
print("BEFORE LoRA")
print("=" * 70)

all_parameters_before = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_before = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print()
print(f"model.parameters() count : {all_parameters_before:,}")
print(f"trainable parameters     : {trainable_before:,}")


# ============================================================
# 5. Prepare model for k-bit training
# ============================================================

print()
print("Preparing model for k-bit training...")

model = prepare_model_for_kbit_training(model)


# ============================================================
# 6. LoRA configuration
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
# 7. Attach LoRA
# ============================================================

print()
print("Attaching LoRA...")

model = get_peft_model(
    model,
    lora_config,
)

print("LoRA attached.")


# ============================================================
# 8. Count parameters AFTER LoRA
# ============================================================

print()
print("=" * 70)
print("AFTER LoRA")
print("=" * 70)

all_parameters_after = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_after = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print()
print(f"model.parameters() count : {all_parameters_after:,}")
print(f"trainable parameters     : {trainable_after:,}")

print()
print("PEFT's parameter report:")

model.print_trainable_parameters()


# ============================================================
# 9. Inspect parameter data types
# ============================================================

print()
print("=" * 70)
print("PARAMETER DATA TYPES")
print("=" * 70)

dtype_counts = {}

for name, parameter in model.named_parameters():

    dtype = str(parameter.dtype)

    if dtype not in dtype_counts:
        dtype_counts[dtype] = 0

    dtype_counts[dtype] += parameter.numel()


for dtype, count in dtype_counts.items():

    print()
    print(f"{dtype}: {count:,} parameters")


# ============================================================
# 10. Inspect LoRA parameters
# ============================================================

print()
print("=" * 70)
print("LORA PARAMETERS")
print("=" * 70)

lora_parameter_count = 0

for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        lora_parameter_count += parameter.numel()

        print(
            f"{name:<80} "
            f"shape={tuple(parameter.shape)}"
        )


print()
print(f"Total trainable LoRA parameters: {lora_parameter_count:,}")


# ============================================================
# 11. GPU memory
# ============================================================

print()
print("=" * 70)
print("GPU MEMORY")
print("=" * 70)

if torch.cuda.is_available():

    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)

    print()
    print(
        f"Allocated: "
        f"{allocated / (1024 ** 3):.2f} GB"
    )

    print(
        f"Reserved:  "
        f"{reserved / (1024 ** 3):.2f} GB"
    )


# ============================================================
# Final
# ============================================================

print()
print("=" * 70)
print("PARAMETER INSPECTION COMPLETE")
print("=" * 70)