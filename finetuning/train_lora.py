"""Hardware-aware LoRA fine-tuning for the local defensive target model."""
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
# The dashboard may choose 128–512 based on available memory.
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


def load_examples(processor, train_file, max_length):
    examples, truncated = [], 0
    for line in train_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        prompt_ids = ids_from_template(processor, row["prompt"], True)
        full_messages = row["prompt"] + row["completion"]
        input_ids = ids_from_template(processor, full_messages, False)
        prompt_length = prompt_ids.shape[-1]
        completion_length = input_ids.shape[-1] - prompt_length
        if completion_length >= max_length:
            raise ValueError("An answer is as long as the full token limit; increase --max-length.")
        prompt_budget = max_length - completion_length
        if prompt_length > prompt_budget:
            # Keep both the start (system guidance) and end (usually the attack text)
            # of the prompt, while always retaining the full expected answer.
            head = min(48, max(1, prompt_budget // 3))
            tail = prompt_budget - head
            prompt_part = input_ids[:, :head]
            if tail:
                prompt_part = torch.cat((prompt_part, input_ids[:, prompt_length - tail:prompt_length]), dim=-1)
            input_ids = torch.cat((prompt_part, input_ids[:, prompt_length:]), dim=-1)
            prompt_length = prompt_budget
            truncated += 1
        labels = input_ids.clone()
        prompt_length = min(prompt_length, input_ids.shape[-1])
        labels[:, :prompt_length] = -100
        if (labels != -100).sum().item() == 0:
            raise ValueError("A training example has no completion tokens after truncation.")
        examples.append((input_ids, labels))
    if not examples:
        raise ValueError("The training file contains no usable examples.")
    return examples, truncated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=0, help="Optional optimizer-step limit; 0 runs all epochs.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                        help="Maximum training tokens per example; lower values use less memory.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-file", default=str(TRAIN_FILE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    train_file = Path(args.train_file)
    output_dir = Path(args.output_dir)
    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not train_file.exists():
        raise SystemExit(f"Training data is missing: {train_file}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this PyTorch installation cannot see a supported NVIDIA GPU.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Loading base checkpoint: {MODEL_ID} on {device.type}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    max_length = max(64, args.max_length)
    examples, truncated_examples = load_examples(processor, train_file, max_length)
    random.Random(42).shuffle(examples)
    print(f"Prepared {len(examples)} examples; max tokens: {max_length}; prompts shortened: {truncated_examples}", flush=True)

    base_model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    base_model.to(device)
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
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=5e-5,
        foreach=False,
    )
    accumulation = 4
    epochs = max(1, args.epochs)
    steps_per_epoch = math.ceil(len(examples) / accumulation)
    total_steps = steps_per_epoch * epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    print(f"Prepared {len(examples)} examples; epochs: {epochs}; optimizer steps: {total_steps}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    step = 0
    stop_training = False
    examples_seen = 0
    for epoch in range(epochs):
        epoch_examples = list(examples)
        random.Random(42 + epoch).shuffle(epoch_examples)
        for index, (input_ids, labels) in enumerate(epoch_examples):
            input_ids, labels = input_ids.to(device), labels.to(device)
            output = model(input_ids=input_ids, labels=labels)
            group_start = (index // accumulation) * accumulation
            group_size = min(accumulation, len(epoch_examples) - group_start)
            (output.loss / group_size).backward()
            loss_total += float(output.loss.detach())
            examples_seen += 1
            is_update = ((index + 1) % accumulation == 0) or (index + 1 == len(epoch_examples))
            if is_update:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                print(f"Epoch {epoch + 1}/{epochs} · step {step}/{total_steps} · mean_loss={loss_total / examples_seen:.4f}", flush=True)
                if args.max_steps > 0 and step >= args.max_steps:
                    stop_training = True
                    break
        if stop_training:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    metadata = {
        "base_model": MODEL_ID,
        "method": "LoRA supervised fine-tuning with automatic CPU/GPU selection and gradient checkpointing",
        "training_examples": len(examples),
        "examples_seen": examples_seen,
        "epochs": epoch + 1,
        "optimizer_steps": step,
        "learning_rate": 5e-5,
        "shuffle_seed": 42,
        "max_sequence_length": max_length,
        "truncated_prompts": truncated_examples,
        "device": device.type,
        "lora_rank": 8,
        "mean_training_loss": loss_total / max(1, examples_seen),
        "created_at": datetime.now().astimezone().isoformat(),
        "adapter_path": str(output_dir),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
