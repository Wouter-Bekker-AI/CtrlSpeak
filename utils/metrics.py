# -*- coding: utf-8 -*-
"""Simple CSV metrics logging for observability."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable

from utils.io_atomic import atomic_write_text

_HEADER = [
    "timestamp_iso",
    "correlation_id",
    "metric",
    "value",
]


class MetricsRecorder:
    """Append structured metrics rows to a CSV file."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self._lock = Lock()
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.target.exists():
            return
        self.target.parent.mkdir(parents=True, exist_ok=True)
        header_row = ",".join(_HEADER)
        atomic_write_text(self.target, f"{header_row}\n")

    def record(self, correlation_id: str, metrics: Dict[str, float | int]) -> None:
        if not metrics:
            return
        timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        rows = []
        for key, value in metrics.items():
            rows.append(
                {
                    "timestamp_iso": timestamp,
                    "correlation_id": correlation_id,
                    "metric": key,
                    "value": value,
                }
            )
        with self._lock:
            write_header = not self.target.exists()
            if write_header:
                self._ensure_header()
            with self.target.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_HEADER)
                if write_header:
                    writer.writeheader()
                for row in rows:
                    writer.writerow(row)


__all__ = ["MetricsRecorder"]
