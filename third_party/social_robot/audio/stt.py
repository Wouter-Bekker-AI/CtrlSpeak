"""Speech-to-text utilities powered by faster-whisper."""

from __future__ import annotations

from typing import Optional

import numpy as np
from faster_whisper import WhisperModel


class FasterWhisperSTT:
    """Wrapper around :mod:`faster_whisper` for low-latency transcription."""

    def __init__(
        self,
        model_size_or_path: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "en",
    ) -> None:
        print(f"-> Loading faster-whisper model '{model_size_or_path}' on {device} ({compute_type}).")
        self.model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type)
        self.language = language

    def run_stt(self, raw_bytes: bytes, sample_rate: int = 16000) -> str:
        if not raw_bytes:
            return ""

        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio_np.size == 0:
            return ""

        segments, _ = self.model.transcribe(
            audio_np,
            language=self.language,
            beam_size=1,
            vad_filter=False,
            suppress_blank=True,
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text


__all__ = ["FasterWhisperSTT"]
