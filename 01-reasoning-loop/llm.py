"""Minimal Ollama client used by the local reasoning-loop lab."""

import os

import requests


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "60"))


def generate(prompt: str, stop=None, temperature: float = 0.2) -> str:
    """Send a prompt to Ollama and return the generated text."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    if stop:
        payload["options"]["stop"] = stop

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure 'ollama serve' is running "
            f"and the model '{OLLAMA_MODEL}' is available."
        ) from exc

    return response.json().get("response", "")
