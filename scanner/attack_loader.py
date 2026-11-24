from pathlib import Path

import yaml


def load_attack_suite(suite_path=None):
    project_folder = Path(__file__).resolve().parent.parent
    payload_file = Path(suite_path) if suite_path else project_folder / "payloads" / "attacks.yaml"
    if not payload_file.is_absolute():
        payload_file = project_folder / payload_file

    if not payload_file.exists():
        raise FileNotFoundError(f"Attack file not found: {payload_file}")

    data = yaml.safe_load(payload_file.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("The YAML file must contain a dictionary.")

    if "canary" not in data:
        raise ValueError("The YAML file is missing the canary value.")

    attacks = data.get("attacks")
    if not isinstance(attacks, list):
        raise ValueError("'attacks' must be a list.")

    required_fields = {
        "id",
        "category",
        "language",
        "prompt",
        "success_criteria",
    }

    for number, attack in enumerate(attacks, start=1):
        if not isinstance(attack, dict):
            raise ValueError(f"Attack number {number} must be a dictionary.")

        missing_fields = required_fields - attack.keys()
        if missing_fields:
            names = ", ".join(sorted(missing_fields))
            raise ValueError(f"Attack number {number} is missing: {names}")

    return data
