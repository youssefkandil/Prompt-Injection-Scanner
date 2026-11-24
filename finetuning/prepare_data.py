"""Build a source-separated SFT dataset from the scanner's attack corpus."""
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
PAYLOADS = ROOT / "payloads" / "attacks.yaml"
DATA_DIR = ROOT / "finetuning" / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_SUITE = ROOT / "finetuning" / "eval_suite.yaml"
MANIFEST = DATA_DIR / "split_manifest.json"
TRAIN_PREFIXES = ("ai-sec-", "gandalf-")
EVAL_PREFIXES = ("deepset-", "hackaprompt-")
SAFE_REFUSAL = (
    "I can't provide hidden system instructions, secret markers, or protected "
    "values. I can still help with a safe request."
)
BENIGN_TRAINING_EXAMPLES = [
    ("What is 18 multiplied by 7? Reply with the number only.", "126"),
    ("Name the capital of Italy. Reply with the city only.", "Rome."),
    ("Translate 'Good night' into French. Reply with the translation only.", "Bonne nuit."),
    ("What is the opposite of 'early'? Reply with one word.", "Late."),
    ("List three common fruits, separated by commas.", "Apples, bananas, oranges."),
    ("Give a simple three-step method for boiling an egg.", "Place the egg in water, bring it to a boil, then simmer until it reaches your preferred firmness."),
    ("Write a polite one-sentence message asking to reschedule a meeting.", "Could we please reschedule our meeting for another time that works for you?"),
    ("What planet is known as the Red Planet? Reply with the name only.", "Mars."),
    ("Explain photosynthesis in one simple sentence.", "Plants use sunlight to turn water and carbon dioxide into sugar and oxygen."),
    ("What is 100 divided by 4? Reply with the number only.", "25"),
    ("Name one renewable energy source.", "Solar energy is a renewable energy source."),
    ("Give a short, friendly greeting for a new coworker.", "Welcome to the team! It is great to have you here."),
    ("What color do you get by mixing blue and yellow paint? Reply with the color only.", "Green."),
    ("Give one practical tip for keeping a desk organized.", "Keep only the items you use daily on the desk and store the rest in labeled drawers."),
    ("Complete the sentence: A week has seven ___. Reply with one word.", "days."),
    ("Write a short thank-you note for a helpful neighbor.", "Thank you for your help today. I really appreciate your kindness."),
    ("Name the largest ocean on Earth.", "The Pacific Ocean."),
    ("Suggest a simple lunch with vegetables and rice.", "Serve cooked rice with sautéed vegetables such as peppers, carrots, and broccoli."),
    ("What is 9 plus 14? Reply with the number only.", "23"),
    ("Explain what a bicycle helmet is for in one sentence.", "A bicycle helmet helps protect the rider's head during a fall or collision."),
]


def main():
    load_dotenv(ROOT / ".env")
    import os

    system_prompt = os.getenv("TARGET_SYSTEM_PROMPT", "").strip()
    if not system_prompt:
        raise SystemExit("TARGET_SYSTEM_PROMPT is missing from the project .env file.")

    suite = yaml.safe_load(PAYLOADS.read_text(encoding="utf-8"))
    canary = suite["canary"]
    train, evaluation, unexpected = [], [], []
    for attack in suite["attacks"]:
        attack_id = attack["id"]
        record = {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": attack["prompt"]},
            ],
            "completion": [
                {"role": "assistant", "content": SAFE_REFUSAL},
            ],
        }
        if attack_id.startswith(TRAIN_PREFIXES):
            train.append(record)
        elif attack_id.startswith(EVAL_PREFIXES):
            evaluation.append(attack)
        else:
            unexpected.append(attack_id)

    if unexpected:
        raise SystemExit("Assign these new attack IDs to train or eval: " + ", ".join(unexpected))
    if not train or not evaluation:
        raise SystemExit("The source-based train/eval split is empty.")
    for user_text, assistant_text in BENIGN_TRAINING_EXAMPLES:
        train.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "completion": [
                {"role": "assistant", "content": assistant_text},
            ],
            "example_type": "benign",
        })
    if any(canary in item["completion"][0]["content"] for item in train):
        raise SystemExit("A training answer contains the protected canary.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TRAIN_FILE.open("w", encoding="utf-8", newline="\n") as output:
        for item in train:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")

    eval_suite = dict(suite)
    eval_suite["suite_name"] = "Held-out source evaluation suite"
    eval_suite["split_note"] = "deepset and hackaprompt sources are excluded from fine-tuning."
    eval_suite["ready_case_count"] = len(evaluation)
    eval_suite["attacks"] = evaluation
    EVAL_SUITE.write_text(yaml.safe_dump(eval_suite, allow_unicode=True, sort_keys=False), encoding="utf-8")

    manifest = {
        "base_suite": str(PAYLOADS.relative_to(ROOT)),
        "train_id_prefixes": list(TRAIN_PREFIXES),
        "evaluation_id_prefixes": list(EVAL_PREFIXES),
        "training_cases": len(train),
        "attack_training_cases": sum(item.get("example_type") != "benign" for item in train),
        "benign_training_cases": sum(item.get("example_type") == "benign" for item in train),
        "held_out_cases": len(evaluation),
        "held_out_ids": [item["id"] for item in evaluation],
        "completion_policy": "Fixed safe refusal; never include the canary in target output.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {TRAIN_FILE.relative_to(ROOT)}: {len(train)} training examples")
    print(f"Created {EVAL_SUITE.relative_to(ROOT)}: {len(evaluation)} held-out attacks")
    print(f"Source split: train={TRAIN_PREFIXES}; evaluation={EVAL_PREFIXES}")


if __name__ == "__main__":
    main()
