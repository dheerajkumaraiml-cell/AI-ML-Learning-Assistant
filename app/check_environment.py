import sys
import importlib.util

import torch
import transformers


print("=" * 70)
print("AI/ML ENVIRONMENT CHECK")
print("=" * 70)


# ============================================================
# Python
# ============================================================

print()
print("Python:")
print(sys.version)


# ============================================================
# PyTorch
# ============================================================

print()
print("PyTorch:")
print(torch.__version__)


# ============================================================
# Transformers
# ============================================================

print()
print("Transformers:")
print(transformers.__version__)


# ============================================================
# PEFT
# ============================================================

print()
print("PEFT:")

if importlib.util.find_spec("peft") is not None:
    import peft
    print(peft.__version__)
else:
    print("NOT INSTALLED")


# ============================================================
# bitsandbytes
# ============================================================

print()
print("bitsandbytes:")

if importlib.util.find_spec("bitsandbytes") is not None:
    import bitsandbytes

    print(
        getattr(
            bitsandbytes,
            "__version__",
            "installed",
        )
    )
else:
    print("NOT INSTALLED")


# ============================================================
# CUDA
# ============================================================

print()
print("CUDA available:")
print(torch.cuda.is_available())


if torch.cuda.is_available():

    print()
    print("PyTorch CUDA version:")
    print(torch.version.cuda)

    print()
    print("GPU:")
    print(torch.cuda.get_device_name(0))

    print()
    print("GPU count:")
    print(torch.cuda.device_count())

    total_memory = (
        torch.cuda.get_device_properties(0).total_memory
    )

    total_gb = total_memory / (1024 ** 3)

    print()
    print("GPU memory:")
    print(f"{total_gb:.2f} GB")

else:

    print()
    print("GPU:")
    print("No CUDA GPU detected.")


print()
print("=" * 70)
print("ENVIRONMENT CHECK COMPLETE")
print("=" * 70)