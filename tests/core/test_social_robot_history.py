from pathlib import Path

import pytest

SOCIAL_ROBOT_DIR = Path(__file__).resolve().parents[2] / "third_party" / "social_robot"
import sys

if str(SOCIAL_ROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(SOCIAL_ROBOT_DIR))

from history_utils import REMOVED_IMAGE_PLACEHOLDER, prune_history_images


@pytest.mark.core_headless
def test_prune_history_images_keeps_latest_image():
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image", "image": "first_image"},
            ],
        },
        {"role": "assistant", "content": "ack"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "latest"},
                {"type": "image", "image": "latest_image"},
            ],
        },
    ]

    prune_history_images(history)

    first_entry_content = history[0]["content"]
    latest_entry_content = history[2]["content"]

    assert all(
        item.get("type") != "image" for item in first_entry_content if isinstance(item, dict)
    )
    assert any(
        item.get("type") == "image" and item.get("image") == "latest_image"
        for item in latest_entry_content
        if isinstance(item, dict)
    )


@pytest.mark.core_headless
def test_prune_history_images_removes_all_when_requested():
    history = [
        {"role": "user", "content": [{"type": "image", "image": "first_image"}]},
        {"role": "assistant", "content": "ack"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "latest"},
                {"type": "image", "image": "latest_image"},
            ],
        },
    ]

    prune_history_images(history, keep_latest=False)

    for entry in history:
        content = entry.get("content")
        if isinstance(content, list):
            assert all(
                item.get("type") != "image"
                for item in content
                if isinstance(item, dict)
            )

    assert history[0]["content"][0]["text"] == REMOVED_IMAGE_PLACEHOLDER
