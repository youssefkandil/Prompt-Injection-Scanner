"""Local OpenAI-compatible endpoint for a base or LoRA-adapted Qwen target."""
import os
import time
import uuid

import torch
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForMultimodalLM, AutoProcessor


MODEL_ID = os.getenv("HF_BASE_MODEL", "Qwen/Qwen3.5-0.8B")
ADAPTER_PATH = os.getenv("HF_ADAPTER_PATH", "").strip()
torch.set_num_threads(max(1, int(os.getenv("HF_CPU_THREADS", "4"))))
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True)
if ADAPTER_PATH:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

app = FastAPI(title="Local Fine-tuned Scanner Target")


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[dict]
    max_tokens: int = 128
    temperature: float = 0


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "adapter_loaded": bool(ADAPTER_PATH)}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    encoded = processor.apply_chat_template(
        request.messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    model_inputs = encoded if isinstance(encoded, dict) else {"input_ids": encoded}
    input_ids = model_inputs["input_ids"]
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=max(1, min(request.max_tokens, 128)),
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )
    answer = processor.decode(
        generated[0, input_ids.shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
    }
