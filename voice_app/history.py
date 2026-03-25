import json
import os
from typing import Dict, List


def load_history(filepath: str) -> List[Dict]:
    """Load conversation history or return an empty list on first run."""

    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def save_history(filepath: str, messages: List[Dict]) -> None:
    """Persist the full message list in OpenAI-compatible JSON format."""

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(messages, file, indent=2, ensure_ascii=False)


def append_message(filepath: str, role: str, content: str, max_messages: int) -> List[Dict]:
    """Append one message, trim old entries, and persist the updated history."""

    messages = load_history(filepath)
    messages.append({"role": role, "content": content})

    if len(messages) > max_messages:
        messages = messages[-max_messages:]

    save_history(filepath, messages)
    return messages
