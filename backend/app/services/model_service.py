
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "qlora_qwen" / "checkpoint-3500"


class ModelService:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. This configuration requires a GPU."
            )

        if not CHECKPOINT.is_dir():
            raise FileNotFoundError(
                f"Adapter checkpoint not found: {CHECKPOINT}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(CHECKPOINT)
        )

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no pad or EOS token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map={"": 0},
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            str(CHECKPOINT),
            is_trainable=False,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, question: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("The model has not been loaded.")

        messages = [
            {"role": "user", "content": question}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )

        inputs = {
            key: value.to("cuda:0")
            for key, value in inputs.items()
        }

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        prompt_length = inputs["input_ids"].shape[1]
        answer_tokens = output_ids[0, prompt_length:]

        return self.tokenizer.decode(
            answer_tokens,
            skip_special_tokens=True,
        ).strip()
