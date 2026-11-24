import os

import httpx
from dotenv import load_dotenv


load_dotenv()


def send_prompt(prompt: str, config: dict | None = None) -> str:
    config = config or {}
    base_url = (config.get("base_url") or os.getenv("TARGET_BASE_URL", "http://127.0.0.1:8001")).rstrip("/")
    endpoint = config.get("endpoint") or os.getenv("TARGET_ENDPOINT", "/chat/completions")
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    target_url = base_url + endpoint
    model_name = config.get("model") or os.getenv("TARGET_MODEL", "demo")
    api_key = config.get("api_key")
    if api_key is None:
        api_key = os.getenv("TARGET_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    messages = []
    system_prompt = (config.get("system_prompt") or os.getenv("TARGET_SYSTEM_PROMPT", "")).strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = httpx.post(
        target_url,
        headers=headers,
        json={"model": model_name, "messages": messages, "temperature": 0, "max_tokens": 128, "reasoning_effort": "none"},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
