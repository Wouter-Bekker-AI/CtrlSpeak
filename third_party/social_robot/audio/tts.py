"""Text-to-speech utilities powered by Kokoro-ONNX."""

from __future__ import annotations

import hashlib
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pyaudio
import requests
from kokoro_onnx import Kokoro, SAMPLE_RATE

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
MODEL_SHA256 = "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"
VOICES_SHA256 = "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"
MODEL_FILENAME = "kokoro-v1.0.onnx"
VOICES_FILENAME = "voices-v1.0.bin"


class KokoroTTS:
    """Generate audio using the Kokoro neural TTS model."""

    def __init__(
        self,
        voice: str = "af_bella",
        speed: float = 1.0,
        model_dir: Optional[Path | str] = None,
    ) -> None:
        warnings.filterwarnings("once", category=UserWarning, module="phonemizer")
        self.voice = voice
        self.speed = speed
        self.sample_rate = SAMPLE_RATE
        self._model_dir = Path(model_dir) if model_dir else Path.home() / ".cache" / "kokoro_onnx"
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model_path = self._model_dir / MODEL_FILENAME
        self._voices_path = self._model_dir / VOICES_FILENAME
        self._ensure_model_files()
        self._engine = Kokoro(str(self._model_path), str(self._voices_path))
        self._validate_voice()
        print("-> Using Kokoro-ONNX for speech synthesis.")
        self._stop_event = threading.Event()
        self._is_playing = False

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def stop_playback(self) -> None:
        self._stop_event.set()

    def _ensure_model_files(self) -> None:
        for url, path, checksum in (
            (MODEL_URL, self._model_path, MODEL_SHA256),
            (VOICES_URL, self._voices_path, VOICES_SHA256),
        ):
            if not path.exists() or self._sha256(path) != checksum:
                self._download_with_checksum(url, path, checksum)

    @staticmethod
    def _sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _download_with_checksum(self, url: str, destination: Path, expected_sha256: str) -> None:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, dir=str(destination.parent)) as tmp_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp_file.write(chunk)
            tmp_path = Path(tmp_file.name)

        actual_sha256 = self._sha256(tmp_path)
        if actual_sha256 != expected_sha256:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {destination.name}: expected {expected_sha256}, got {actual_sha256}"
            )

        tmp_path.replace(destination)

    def _validate_voice(self) -> None:
        available = self._engine.get_voices()
        if self.voice not in available:
            raise ValueError(f"Voice '{self.voice}' is not available. Choose from: {', '.join(available)}")

    def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        audio, sample_rate = self._engine.create(text, voice=self.voice, speed=self.speed)
        self.sample_rate = sample_rate
        return audio.astype(np.float32, copy=False)

    def _chunk_audio(self, audio_data: np.ndarray, chunk_size: int) -> List[np.ndarray]:
        return [audio_data[i : i + chunk_size] for i in range(0, len(audio_data), chunk_size)]

    def play_audio_with_amplitude(
        self,
        audio_data: np.ndarray,
        amplitude_callback: Optional[Callable[[float], None]] = None,
        chunk_duration: float = 0.02,
    ) -> None:
        if audio_data is None or len(audio_data) == 0:
            if amplitude_callback:
                amplitude_callback(0.0)
            return

        self._stop_event.clear()
        self._is_playing = True

        audio_float = audio_data.astype(np.float32, copy=False)
        chunk_size = max(1, int(self.sample_rate * chunk_duration))
        chunks = self._chunk_audio(audio_float, chunk_size)
        if not chunks:
            self._is_playing = False
            return

        rms_values = [float(np.sqrt(np.mean(np.square(chunk)) + 1e-8)) for chunk in chunks]
        max_rms = max(rms_values) or 1.0
        normalized_levels = [min(rms / max_rms, 1.0) for rms in rms_values]

        audio_int16 = np.clip(audio_float * 32767.0, -32768, 32767).astype(np.int16)

        pa = pyaudio.PyAudio()
        stream = None

        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=4096,
            )

            cursor = 0
            for chunk, level in zip(chunks, normalized_levels):
                if self._stop_event.is_set():
                    break
                frames = audio_int16[cursor : cursor + len(chunk)]
                stream.write(frames.tobytes())
                cursor += len(chunk)
                if amplitude_callback:
                    amplitude_callback(level)
        finally:
            self._is_playing = False
            if stream is not None:
                try:
                    output_latency = stream.get_output_latency()
                except Exception:
                    output_latency = 0.0

                drain_wait = (
                    max(output_latency, chunk_duration)
                    if output_latency and output_latency > 0
                    else chunk_duration
                )
                if drain_wait and drain_wait > 0:
                    time.sleep(drain_wait)

                if amplitude_callback:
                    amplitude_callback(0.0)
                stream.stop_stream()
                stream.close()
            elif amplitude_callback:
                amplitude_callback(0.0)
            pa.terminate()

    def available_voices(self) -> Tuple[str, ...]:
        return tuple(self._engine.get_voices())


__all__ = ["KokoroTTS"]
