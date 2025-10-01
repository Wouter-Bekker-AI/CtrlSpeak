"""Single-image persistence helpers for bot identities."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.config_paths import get_logger
from utils.memory_paths import get_bot_screenshots_dir


logger = get_logger(__name__)

_IMAGE_FILENAME = "current.png"
_METADATA_FILENAME = "metadata.json"


@dataclass(slots=True)
class IdentityImageRecord:
    """Description of the most recent image stored for an identity."""

    path: Path
    image_b64: str
    source: Optional[str]
    updated_at: datetime

    @property
    def updated_at_iso(self) -> str:
        return self.updated_at.replace(tzinfo=timezone.utc).isoformat()

    def as_metadata(self, *, vision_request: Optional[str] = None) -> dict[str, str]:
        metadata: dict[str, str] = {
            "vision_file": str(self.path),
            "vision_updated": self.updated_at_iso,
        }
        if self.source:
            metadata["vision_source"] = self.source
        if vision_request:
            metadata["vision_request"] = vision_request
        metadata["vision_attached"] = "true"
        return metadata


def _metadata_path(identity: str) -> Path:
    return get_bot_screenshots_dir(identity) / _METADATA_FILENAME


def _image_path(identity: str) -> Path:
    return get_bot_screenshots_dir(identity) / _IMAGE_FILENAME


def write_identity_image_from_base64(
    identity: str,
    image_b64: str,
    *,
    source: Optional[str] = None,
) -> Optional[IdentityImageRecord]:
    """Persist ``image_b64`` as the identity's single stored PNG."""

    try:
        image_bytes = base64.b64decode(image_b64.encode("ascii"), validate=True)
    except Exception:
        logger.exception("Failed to decode base64 payload for identity image %s", identity)
        return None
    return write_identity_image(identity, image_bytes, source=source)


def write_identity_image(
    identity: str,
    image_bytes: bytes,
    *,
    source: Optional[str] = None,
) -> Optional[IdentityImageRecord]:
    """Write ``image_bytes`` as the current PNG for ``identity``."""

    directory = get_bot_screenshots_dir(identity)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to create image directory for %s", identity)
        return None

    target = _image_path(identity)
    temp_path = target.with_suffix(".tmp")
    try:
        temp_path.write_bytes(image_bytes)
        temp_path.replace(target)
    except Exception:
        logger.exception("Failed to persist image for identity %s", identity)
        temp_path.unlink(missing_ok=True)
        return None
    finally:
        for extra in directory.glob("*.png"):
            if extra != target:
                extra.unlink(missing_ok=True)

    metadata = {
        "updated_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    }
    if source:
        metadata["source"] = source
    try:
        _metadata_path(identity).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("Failed to persist image metadata for %s", identity)

    return load_identity_image(identity)


def load_identity_image(identity: str) -> Optional[IdentityImageRecord]:
    """Return the stored image for ``identity`` if present."""

    target = _image_path(identity)
    if not target.exists():
        return None

    try:
        image_bytes = target.read_bytes()
    except Exception:
        logger.exception("Failed to read stored image for %s", identity)
        return None

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    metadata_path = _metadata_path(identity)
    source: Optional[str] = None
    updated_at = None
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = payload.get("source")
            updated_raw = payload.get("updated_at")
            if isinstance(updated_raw, str):
                updated_at = datetime.fromisoformat(updated_raw)
        except Exception:
            logger.exception("Failed to parse image metadata for %s", identity)
    if updated_at is None:
        updated_at = datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc)

    return IdentityImageRecord(target, image_b64, source, updated_at)


def clear_identity_image(identity: str) -> None:
    """Remove the stored PNG and metadata for ``identity``."""

    _image_path(identity).unlink(missing_ok=True)
    _metadata_path(identity).unlink(missing_ok=True)


def is_image_request(text: str) -> bool:
    """Return ``True`` when ``text`` appears to reference the stored image."""

    if not text:
        return False
    lowered = text.lower()
    keywords = ("image", "picture", "photo", "screenshot", "screen", "clipboard")
    return any(token in lowered for token in keywords)


__all__ = [
    "IdentityImageRecord",
    "clear_identity_image",
    "is_image_request",
    "load_identity_image",
    "write_identity_image",
    "write_identity_image_from_base64",
]

