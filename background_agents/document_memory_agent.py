"""Stubbed document memory agent.

CtrlSpeak previously ingested bundled Markdown documentation into a
per-identity ChromaDB collection so the personas could retrieve the
chunks during conversation. We are preparing to migrate to the
third-party Docling RAG agent, so the legacy ingestion workflow is now
disabled. This module keeps the public refresh function available for
callers but turns it into a no-op that simply records that the refresh
was skipped.
"""
from __future__ import annotations

from typing import List

from utils.config_paths import get_logger

_LOGGER = get_logger(__name__)


def refresh_document_memory(
    identity: str,
    *,
    force: bool = False,
    reason: str | None = None,
) -> bool:
    """Skip documentation ingestion for ``identity``.

    The previous implementation populated a Chroma-backed vector store
    with Markdown documentation. That pipeline has been retired while we
    prepare the switch to the Docling RAG agent, so refresh requests now
    log the skip and succeed immediately.
    """

    details: List[str] = []
    if force:
        details.append("force=True")
    if reason:
        details.append(f"reason={reason}")
    suffix = f" ({', '.join(details)})" if details else ""

    message = (
        f"[DocMemory] Documentation ingestion disabled; skipping refresh for '{identity}'."
    )
    print(message)
    _LOGGER.info(
        "Documentation refresh skipped for identity '%s'%s; ingestion disabled.",
        identity,
        suffix,
    )
    return True


__all__ = ["refresh_document_memory"]
