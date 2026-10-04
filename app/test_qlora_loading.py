import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


print("=" * 70)
print("QLORA 4-BIT MODEL LOADING TEST")
print("=" * 70)


# ============================================================
# 1. Check CUDA
# ============================================================

print()
print("CUDA available:")
print(torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. QLoRA test requires a CUDA GPU."
    )


print()
print("GPU:")
print(torch.cuda.get_device_name(0))


# ============================================================
# 2. Configure 4-bit quantization
# ============================================================

print()
print("=" * 70)
print("CONFIGURING 4-BIT QUANTIZATION")
print("=" * 70)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print()
print("4-bit loading : True")
print("Quantization  : NF4")
print("Compute dtype : float16")
print("Double quant  : True")


# ============================================================
# 3. Load tokenizer
# ============================================================

print()
print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded.")


# ============================================================
# 4. Load model
# ============================================================

print()
print("=" * 70)
print("LOADING MODEL IN 4-BIT")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
)

print()
print("Model loaded successfully.")


# ============================================================
# 5. Inspect model device
# ============================================================

print()
print("=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print()
print("Model class:")
print(type(model).__name__)

print()
print("Model device:")
print(model.device)


# ============================================================
# 6. GPU memory usage
# ============================================================

allocated = torch.cuda.memory_allocated(0)
reserved = torch.cuda.memory_reserved(0)

allocated_gb = allocated / (1024 ** 3)
reserved_gb = reserved / (1024 ** 3)

print()
print("GPU memory allocated:")
print(f"{allocated_gb:.2f} GB")

print()
print("GPU memory reserved:")
print(f"{reserved_gb:.2f} GB")


# ============================================================
# 7. Parameter information
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print()
print("Total model parameters:")
print(f"{total_parameters:,}")


# ============================================================
# 8. Simple inference test
# ============================================================

print()
print("=" * 70)
print("SIMPLE INFERENCE TEST")
print("=" * 70)

messages = [
    {
        "role": "user",
        "content": "What is machine learning?",
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to(model.device)


print()
print("Generating a short response...")

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
    )


generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=False,
)

print()
print("Generated text:")
print(generated_text)


# ============================================================
# Final
# ============================================================

print()
print("=" * 70)
print("QLORA LOADING TEST COMPLETE")
print("=" * 70)