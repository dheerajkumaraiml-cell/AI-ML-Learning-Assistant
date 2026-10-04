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


print("=" * 70)
print("QLORA SETUP TEST")
print("=" * 70)


# ============================================================
# 1. Check CUDA
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required.")

print()
print("GPU:")
print(torch.cuda.get_device_name(0))


# ============================================================
# 2. 4-bit quantization configuration
# ============================================================

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 3. Load tokenizer
# ============================================================

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded.")


# ============================================================
# 4. Load quantized model
# ============================================================

print()
print("Loading 4-bit model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
)

print("4-bit model loaded.")


# ============================================================
# 5. Prepare model for k-bit training
# ============================================================

print()
print("Preparing model for k-bit training...")

model = prepare_model_for_kbit_training(model)

print("Model prepared.")


# ============================================================
# 6. LoRA configuration
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


print()
print("LoRA configuration:")
print(f"  Rank (r)       : {lora_config.r}")
print(f"  Alpha          : {lora_config.lora_alpha}")
print(f"  Dropout        : {lora_config.lora_dropout}")
print(f"  Target modules : {lora_config.target_modules}")


# ============================================================
# 7. Attach LoRA adapters
# ============================================================

print()
print("Attaching LoRA adapters...")

model = get_peft_model(
    model,
    lora_config,
)

print("LoRA adapters attached.")


# ============================================================
# 8. Show trainable parameters
# ============================================================

print()
print("=" * 70)
print("PARAMETER SUMMARY")
print("=" * 70)

model.print_trainable_parameters()


# ============================================================
# 9. Calculate parameter counts ourselves
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

trainable_percentage = (
    trainable_parameters / total_parameters
) * 100


print()
print("Total parameters:")
print(f"{total_parameters:,}")

print()
print("Trainable parameters:")
print(f"{trainable_parameters:,}")

print()
print("Trainable percentage:")
print(f"{trainable_percentage:.4f}%")


# ============================================================
# 10. GPU memory
# ============================================================

allocated = torch.cuda.memory_allocated(0)
reserved = torch.cuda.memory_reserved(0)

print()
print("GPU memory allocated:")
print(f"{allocated / (1024 ** 3):.2f} GB")

print()
print("GPU memory reserved:")
print(f"{reserved / (1024 ** 3):.2f} GB")


# ============================================================
# Final
# ============================================================

print()
print("=" * 70)
print("QLORA SETUP TEST COMPLETE")
print("=" * 70)