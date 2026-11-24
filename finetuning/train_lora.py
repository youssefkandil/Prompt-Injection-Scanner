"""CPU LoRA supervised fine-tuning for the local defensive target model."""
import argparse
import json
import os
import math
import random
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = os.getenv("HF_BASE_MODEL", "Qwen/Qwen3.5-0.8B")
TRAIN_FILE = ROOT / "finetuning" / "data" / "train.jsonl"
OUTPUT_DIR = ROOT / "finetuning" / "output" / "qwen3.5-0.8b-lora"
MAX_LENGTH = 256


def ids_from_template(processor, messages, add_generation_prompt):
    encoded = processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    return encoded["input_ids"] if isinstance(encoded, dict) else encoded


def load_examples(processor):
    examples = []
    for line in TRAIN_FILE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        prompt_ids = ids_from_template(processor, row["prompt"], True)
        full_messages = row["prompt"] + row["completion"]
        input_ids = ids_from_template(processor, full_messages, False)
        prompt_length = prompt_ids.shape[-1]
        if input_ids.shape[-1] > MAX_LENGTH:
            offset = input_ids.shape[-1] - MAX_LENGTH
            input_ids = input_ids[:, -MAX_LENGTH:]
            prompt_length = max(0, prompt_length - offset)
        labels = input_ids.clone()
        prompt_length = min(prompt_length, input_ids.shape[-1])
        labels[:, :prompt_length] = -100
        if (labels != -100).sum().item() == 0:
            raise ValueError("A training example has no completion tokens after truncation.")
        examples.append((input_ids, labels))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=0, help="Limit optimizer steps for a diagnostic run; 0 trains one epoch.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-file", default=str(TRAIN_FILE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    train_file = Path(args.train_file)
    output_dir = Path(args.output_dir)
    torch.set_num_threads(max(1, args.threads))
    if not train_file.exists():
        raise SystemExit(f"Training data is missing: {train_file}")

    print(f"Loading base checkpoint: {MODEL_ID}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    examples = []
    for line in train_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        prompt_ids = ids_from_template(processor, row["prompt"], True)
        full_messages = row["prompt"] + row["completion"]
        input_ids = ids_from_template(processor, full_messages, False)
        prompt_length = prompt_ids.shape[-1]
        if input_ids.shape[-1] > MAX_LENGTH:
            offset = input_ids.shape[-1] - MAX_LENGTH
            input_ids = input_ids[:, -MAX_LENGTH:]
            prompt_length = max(0, prompt_length - offset)
        labels = input_ids.clone()
        prompt_length = min(prompt_length, input_ids.shape[-1])
        labels[:, :prompt_length] = -100
        if (labels != -100).sum().item() == 0:
            raise ValueError("A training example has no completion tokens after truncation.")
        examples.append((input_ids, labels))
    random.Random(42).shuffle(examples)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=5e-5,
    )
    accumulation = 4
    total_steps = math.ceil(len(examples) / accumulation)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    print(f"Prepared {len(examples)} examples; optimizer steps: {total_steps}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    step = 0
    for index, (input_ids, labels) in enumerate(examples):
        output = model(input_ids=input_ids, labels=labels)
        (output.loss / accumulation).backward()
        loss_total += float(output.loss.detach())
        is_update = ((index + 1) % accumulation == 0) or (index + 1 == len(examples))
        if is_update:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            print(f"Step {step}/{total_steps} | mean_loss={loss_total / (index + 1):.4f}", flush=True)
            if args.max_steps > 0 and step >= args.max_steps:
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    metadata = {
        "base_model": MODEL_ID,
        "method": "LoRA supervised fine-tuning (CPU, standard causal cross-entropy)",
        "training_examples": len(examples),
        "epochs": "partial diagnostic run" if args.max_steps > 0 else 1,
        "optimizer_steps": step,
        "learning_rate": 5e-5,
        "shuffle_seed": 42,
        "mean_training_loss": loss_total / max(1, len(examples)),
        "created_at": datetime.now().astimezone().isoformat(),
        "adapter_path": str(output_dir),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
