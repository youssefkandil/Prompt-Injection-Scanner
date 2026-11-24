"""Small local Qwen target used by the Streamlit demonstration."""
from functools import lru_cache
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-0.8B"
ADAPTER_DIR = ROOT / "finetuning" / "output" / "streamlit-adapter"


@lru_cache(maxsize=2)
def load_model(adapter_path=""):
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True
    )
    if adapter_path and (Path(adapter_path) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return processor, model


def generate(prompt, system_prompt, use_adapter=True):
    adapter_path = str(ADAPTER_DIR) if use_adapter and (ADAPTER_DIR / "adapter_config.json").exists() else ""
    processor, model = load_model(adapter_path)
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})
    encoded = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", enable_thinking=False,
    )
    model_inputs = encoded if isinstance(encoded, dict) else {"input_ids": encoded}
    input_ids = model_inputs["input_ids"]
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs, do_sample=False, max_new_tokens=128,
            pad_token_id=processor.tokenizer.eos_token_id, use_cache=True,
        )
    return processor.decode(generated[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()


def clear_model_cache():
    load_model.cache_clear()
