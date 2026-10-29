
# import json
# from pathlib import Path

# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
# )
# from peft import PeftModel


# # ============================================================
# # Configuration
# # ============================================================

# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# CHECKPOINT = Path("checkpoints/qlora_qwen/checkpoint-3500")
# TEST_FILE = Path("dataset/splits/test.jsonl")
# OUTPUT_FILE = Path("results/generation_comparison.jsonl")

# NUM_EXAMPLES = 20
# MAX_NEW_TOKENS = 256
# MAX_INPUT_TOKENS = 512

# # Fixed seed for reproducibility.
# SEED = 42


# # ============================================================
# # Data
# # ============================================================

# def load_test_examples(path):
#     if not path.is_file():
#         raise FileNotFoundError(
#             f"Test file not found: {path}"
#         )

#     examples = []

#     with open(path, "r", encoding="utf-8") as file:
#         for line_number, line in enumerate(file, start=1):
#             if not line.strip():
#                 continue

#             try:
#                 example = json.loads(line)
#             except json.JSONDecodeError as exc:
#                 raise ValueError(
#                     f"Invalid JSON at line {line_number}"
#                 ) from exc

#             if not example.get("question") or not example.get("answer"):
#                 continue

#             examples.append(example)

#     if not examples:
#         raise ValueError("No usable test examples were found.")

#     # Keep the selection deterministic.
#     return examples[:min(NUM_EXAMPLES, len(examples))]


# # ============================================================
# # Model loading
# # ============================================================

# def load_model_and_tokenizer(use_adapter):
#     print("\nLoading tokenizer...")

#     tokenizer_source = (
#         str(CHECKPOINT) if use_adapter else MODEL_NAME
#     )

#     tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_source
#     )

#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#     )

#     print(
#         "Loading "
#         + ("fine-tuned model..." if use_adapter else "base model...")
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=quantization_config,
#         device_map={"": 0},
#     )

#     if use_adapter:
#         model = PeftModel.from_pretrained(
#             model,
#             CHECKPOINT,
#             is_trainable=False,
#         )

#     model.eval()

#     return model, tokenizer


# # ============================================================
# # Generation
# # ============================================================

# @torch.inference_mode()
# def generate_answer(model, tokenizer, question):
#     messages = [
#         {
#             "role": "user",
#             "content": question,
#         }
#     ]

#     # Use the same chat-template approach as training.
#     prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=MAX_INPUT_TOKENS,
#         add_special_tokens=False,
#     )

#     # The model is loaded on GPU 0.
#     inputs = {
#         key: value.to("cuda:0")
#         for key, value in inputs.items()
#     }

#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=MAX_NEW_TOKENS,
#         do_sample=False,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         use_cache=True,
#     )

#     # Decode only newly generated tokens, not the prompt.
#     new_tokens = output_ids[
#         :, inputs["input_ids"].shape[1]:
#     ]

#     answer = tokenizer.decode(
#         new_tokens[0],
#         skip_special_tokens=True,
#     ).strip()

#     return answer


# # ============================================================
# # Main evaluation
# # ============================================================

# def main():
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA GPU is required.")

#     if not CHECKPOINT.is_dir():
#         raise FileNotFoundError(
#             f"Checkpoint not found: {CHECKPOINT}"
#         )

#     if not (CHECKPOINT / "adapter_config.json").is_file():
#         raise FileNotFoundError(
#             "The checkpoint does not contain adapter_config.json."
#         )

#     torch.manual_seed(SEED)

#     print("=" * 70)
#     print("BASE MODEL VS FINE-TUNED MODEL")
#     print("=" * 70)
#     print(f"GPU: {torch.cuda.get_device_name(0)}")

#     examples = load_test_examples(TEST_FILE)

#     print(f"Selected test examples: {len(examples)}")

#     results = []

#     # --------------------------------------------------------
#     # Evaluate base model, then release it from GPU memory.
#     # --------------------------------------------------------

#     base_model, base_tokenizer = load_model_and_tokenizer(
#         use_adapter=False
#     )

#     for index, example in enumerate(examples, start=1):
#         answer = generate_answer(
#             base_model,
#             base_tokenizer,
#             example["question"],
#         )

#         results.append({
#             "index": index,
#             "question": example["question"],
#             "reference_answer": example["answer"],
#             "topic": example.get("topic"),
#             "base_answer": answer,
#         })

#         if index % 5 == 0:
#             print(f"Base model: {index}/{len(examples)}")

#     del base_model
#     del base_tokenizer
#     del answer

#     import gc
#     gc.collect()
#     torch.cuda.empty_cache()

#     # --------------------------------------------------------
#     # Evaluate fine-tuned model on the exact same questions.
#     # --------------------------------------------------------

#     tuned_model, tuned_tokenizer = load_model_and_tokenizer(
#         use_adapter=True
#     )

#     for index, result in enumerate(results, start=1):
#         result["fine_tuned_answer"] = generate_answer(
#             tuned_model,
#             tuned_tokenizer,
#             result["question"],
#         )

#         if index % 5 == 0:
#             print(f"Fine-tuned model: {index}/{len(results)}")

#     # --------------------------------------------------------
#     # Save results without modifying training data/checkpoints.
#     # --------------------------------------------------------

#     OUTPUT_FILE.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     with open(
#         OUTPUT_FILE,
#         "w",
#         encoding="utf-8",
#     ) as file:
#         for result in results:
#             file.write(
#                 json.dumps(
#                     result,
#                     ensure_ascii=False,
#                 )
#                 + "\n"
#             )

#     print("\n" + "=" * 70)
#     print("GENERATION EVALUATION COMPLETE")
#     print("=" * 70)
#     print(f"Examples compared: {len(results)}")
#     print(f"Results saved to: {OUTPUT_FILE}")
#     print("Review the generated answers for correctness and relevance.")

#     del tuned_model
#     del tuned_tokenizer
#     gc.collect()
#     torch.cuda.empty_cache()


# if __name__ == "__main__":
#     main()













import gc
import json
from pathlib import Path

import torch
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

CHECKPOINT = Path("checkpoints/qlora_qwen/checkpoint-3500")

# Stratified 50-question sample created from the held-out test set.
TEST_FILE = Path("results/evaluation_sample_50.jsonl")

OUTPUT_FILE = Path("results/generation_comparison_50.jsonl")

MAX_EXAMPLES = 50
MAX_NEW_TOKENS = 256
MAX_INPUT_TOKENS = 512

SEED = 42


# ============================================================
# Data loading and validation
# ============================================================

def load_test_examples(path):
    """Load and validate the preselected evaluation questions."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation sample not found: {path}\n"
            "Create results/evaluation_sample_50.jsonl first."
        )

    examples = []

    with open(path, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}"
                ) from exc

            question = example.get("question")
            reference_answer = example.get("answer")

            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Missing or invalid question at line {line_number}"
                )

            if (
                not isinstance(reference_answer, str)
                or not reference_answer.strip()
            ):
                raise ValueError(
                    f"Missing or invalid answer at line {line_number}"
                )

            examples.append(example)

    if not examples:
        raise ValueError("No usable evaluation examples were found.")

    if len(examples) != MAX_EXAMPLES:
        raise ValueError(
            f"Expected exactly {MAX_EXAMPLES} examples, "
            f"but found {len(examples)}."
        )

    # The sample-generation script selects questions from the held-out
    # test split. Verify categories so an accidental file mix-up is visible.
    categories = {}
    for example in examples:
        category = example.get("category", "Unknown")
        categories[category] = categories.get(category, 0) + 1

    print("\nEvaluation sample category distribution:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    return examples


# ============================================================
# Model loading
# ============================================================

def load_model_and_tokenizer(use_adapter):
    """Load the base model, optionally attaching the trained LoRA adapter."""

    tokenizer_source = str(CHECKPOINT) if use_adapter else MODEL_NAME

    print(f"\nLoading tokenizer from: {tokenizer_source}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither a pad token nor an EOS token."
            )
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_label = "fine-tuned model" if use_adapter else "base model"
    print(f"Loading {model_label}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map={"": 0},
    )

    if use_adapter:
        model = PeftModel.from_pretrained(
            model,
            CHECKPOINT,
            is_trainable=False,
        )

    model.eval()

    return model, tokenizer


# ============================================================
# Generation
# ============================================================

@torch.inference_mode()
def generate_answer(model, tokenizer, question):
    """Generate an answer using the same prompt and decoding settings."""

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        add_special_tokens=False,
    )

    # This script intentionally loads the model on CUDA device 0.
    inputs = {
        key: value.to("cuda:0")
        for key, value in inputs.items()
    }

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # Only decode newly generated tokens, excluding the input prompt.
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, prompt_length:]

    answer = tokenizer.decode(
        new_tokens[0],
        skip_special_tokens=True,
    ).strip()

    return answer


# ============================================================
# Main evaluation
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this evaluation.")

    if not CHECKPOINT.is_dir():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    if not (CHECKPOINT / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No adapter_config.json found in {CHECKPOINT}"
        )

    torch.manual_seed(SEED)

    print("=" * 70)
    print("BASE MODEL VS FINE-TUNED MODEL")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Evaluation file: {TEST_FILE}")

    examples = load_test_examples(TEST_FILE)
    print(f"\nSelected test examples: {len(examples)}")

    results = []

    # --------------------------------------------------------
    # Evaluate base model.
    # --------------------------------------------------------

    base_model, base_tokenizer = load_model_and_tokenizer(
        use_adapter=False
    )

    try:
        for index, example in enumerate(examples, start=1):
            answer = generate_answer(
                base_model,
                base_tokenizer,
                example["question"],
            )

            results.append({
                "index": index,
                "question": example["question"],
                "reference_answer": example["answer"],
                "topic": example.get("topic"),
                "category": example.get("category"),
                "level": example.get("level"),
                "base_answer": answer,
            })

            if index % 5 == 0 or index == len(examples):
                print(f"Base model: {index}/{len(examples)}")

    finally:
        del base_model
        del base_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Evaluate fine-tuned model on exactly the same questions.
    # --------------------------------------------------------

    tuned_model, tuned_tokenizer = load_model_and_tokenizer(
        use_adapter=True
    )

    try:
        for index, result in enumerate(results, start=1):
            result["fine_tuned_answer"] = generate_answer(
                tuned_model,
                tuned_tokenizer,
                result["question"],
            )

            if index % 5 == 0 or index == len(results):
                print(
                    f"Fine-tuned model: {index}/{len(results)}"
                )

    finally:
        del tuned_model
        del tuned_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Save comparison results.
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    print("\n" + "=" * 70)
    print("GENERATION EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Examples compared: {len(results)}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("Each record contains both model answers and the reference answer.")


if __name__ == "__main__":
    main()
