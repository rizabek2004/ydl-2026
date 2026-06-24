"""gemma4 chat client (OpenAI-compatible chat-completions API on llm.alem.ai)."""
import time

import requests

import config


def chat(messages, temperature: float = 0.1, timeout: int = 60, retries: int = 5) -> str:
    """Send chat messages to gemma4 and return the assistant's text.

    Low default temperature keeps answers grounded (anti-hallucination support).
    Retries with exponential backoff on 429 (rate limit) / 5xx.
    `messages` is a list of {"role": ..., "content": ...} dicts.
    """
    headers = {
        "Authorization": f"Bearer {config.require_chat_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    delay = 2.0
    for attempt in range(retries):
        resp = requests.post(
            config.CHAT_URL, headers=headers, json=payload, timeout=timeout
        )
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
