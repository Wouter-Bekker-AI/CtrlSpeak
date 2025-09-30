from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

REMOVED_IMAGE_PLACEHOLDER = "[previous image removed]"


def load_history(memory_path: Optional[Path]) -> List[dict]:
    if memory_path:
        history_file = memory_path / "conversation.json"
        if history_file.exists():
            try:
                return json.loads(history_file.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"-> Failed to load conversation history: {exc}")
    return []


def save_history(memory_path: Optional[Path], history: List[dict]) -> None:
    if memory_path:
        try:
            history_file = memory_path / "conversation.json"
            history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"-> Failed to save conversation history: {exc}")


def _entry_contains_image(entry: dict) -> bool:
    content = entry.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image" and item.get("image"):
                return True
    return False


def prune_history_images(history: List[dict], *, keep_latest: bool = True) -> None:
    """Remove base64 image payloads from all but the most recent vision entry."""

    latest_index: Optional[int] = None
    if keep_latest:
        for index in range(len(history) - 1, -1, -1):
            if _entry_contains_image(history[index]):
                latest_index = index
                break

    for index, entry in enumerate(history):
        content = entry.get("content")
        if not isinstance(content, list):
            continue

        new_content: list[dict] = []
        removed_image = False
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                if keep_latest and index == latest_index:
                    new_content.append(item)
                else:
                    removed_image = True
                continue
            new_content.append(item)

        if not new_content and removed_image:
            new_content = [{"type": "text", "text": REMOVED_IMAGE_PLACEHOLDER}]
        elif removed_image:
            text_items = [i for i in new_content if isinstance(i, dict) and i.get("type") == "text"]
            if not text_items:
                new_content.append({"type": "text", "text": REMOVED_IMAGE_PLACEHOLDER})

        if removed_image:
            entry["content"] = new_content
