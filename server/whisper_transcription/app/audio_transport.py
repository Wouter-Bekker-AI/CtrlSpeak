"""Lossless sender preparation shared by desktop, gateway and external clients.

No GUI, model or credential imports. Temporary files belong to this context only.
Already compressed audio is never re-encoded. PCM16 -> FLAC changes no samples.
"""
from __future__ import annotations

import io
import logging
import mimetypes
import tempfile
import time
import wave
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

RELEASE_VERSION = "0.7.5"
TRANSPORT_PROTOCOL = 1
MAX_AUDIO_BYTES = 100 * 1024 * 1024
MAX_AUDIO_SECONDS = 1800
LOGGER = logging.getLogger("ctrlspeak.audio_transport")


class AudioLimitError(ValueError):
    """Malformed audio or a bounded decoding limit was reached."""


@dataclass(frozen=True)
class PreparedAudio:
    path: Path
    filename: str
    content_type: str
    original_bytes: int
    upload_bytes: int
    outcome: str
    preparation_ms: float

    def telemetry(self) -> dict:
        return {k: v for k, v in asdict(self).items()
                if k not in {"path", "filename"}}


def audio_format(path: Path) -> tuple[str, str]:
    """Prefer actual container bytes over misleading file extensions."""
    with path.open("rb") as source:
        head = source.read(64)
    if head.startswith(b"fLaC"):
        return ".flac", "audio/flac"
    if head.startswith(b"OggS"):
        return ".ogg", "audio/ogg"
    if head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
        return ".wav", "audio/wav"
    if head.startswith(b"ID3") or (len(head) > 1 and head[0] == 255
                                     and head[1] & 224 == 224 and head[1] & 6):
        return ".mp3", "audio/mpeg"
    if head[4:8] == b"ftyp":
        return ".m4a", "audio/mp4"
    suffix = path.suffix.lower()
    return suffix, mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def capabilities() -> dict:
    return {"protocol": TRANSPORT_PROTOCOL, "lossless_encoding": "flac",
            "accepted_formats": ["wav", "flac", "ogg", "mp3", "m4a", "webm"],
            "compressed_passthrough": True, "max_upload_bytes": MAX_AUDIO_BYTES,
            "max_decoded_pcm16_bytes": MAX_AUDIO_BYTES,
            "max_duration_seconds": MAX_AUDIO_SECONDS}


@contextmanager
def prepare_audio(path: Path, *, accept_flac: bool = True,
                  timeout_seconds: float = 10, cancelled=None) -> Iterator[PreparedAudio]:
    path = Path(path)
    started = time.monotonic()
    size = path.stat().st_size
    if size > MAX_AUDIO_BYTES:
        raise AudioLimitError("audio exceeds upload limit")
    suffix, mime = audio_format(path)

    def result(selected: Path, extension: str, content_type: str, outcome: str):
        return PreparedAudio(selected, "recording" + extension, content_type, size,
                             selected.stat().st_size, outcome,
                             round((time.monotonic() - started) * 1000, 3))

    if suffix != ".wav" or not accept_flac:
        yield result(path, suffix, mime, "compressed_passthrough" if suffix in
                     {".flac", ".ogg", ".mp3", ".m4a", ".webm"} else "legacy_passthrough")
        return
    with tempfile.TemporaryDirectory(prefix="ctrlspeak-transport-") as directory:
        target = Path(directory) / "recording.flac"
        outcome = "unsupported_pcm_passthrough"
        selected = None
        try:
            import av
            import numpy as np

            with wave.open(str(path), "rb") as source:
                channels, rate = source.getnchannels(), source.getframerate()
                frames = source.getnframes()
                if (source.getsampwidth() != 2 or channels not in (1, 2)
                        or source.getcomptype() != "NONE"):
                    raise ValueError("only integer PCM16 mono/stereo is encoded")
                if frames * channels * 2 > MAX_AUDIO_BYTES or frames / rate > MAX_AUDIO_SECONDS:
                    raise AudioLimitError("audio exceeds lossless preparation limits")
                if not frames:
                    raise ValueError("empty audio")
                layout = "mono" if channels == 1 else "stereo"
                count = 0
                with av.open(str(target), "w", format="flac") as container:
                    stream = container.add_stream("flac", rate=rate)
                    stream.layout = layout
                    stream.codec_context.format = "s16"
                    stream.codec_context.options = {"compression_level": "5"}
                    while data := source.readframes(4096):
                        if cancelled is not None and cancelled.is_set():
                            raise InterruptedError("audio preparation cancelled")
                        if time.monotonic() - started > timeout_seconds:
                            raise TimeoutError("audio preparation exceeded deadline")
                        frame = av.AudioFrame.from_ndarray(
                            np.frombuffer(data, dtype="<i2").reshape(1, -1),
                            format="s16", layout=layout)
                        frame.sample_rate = rate
                        frame.pts = count
                        count += frame.samples
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode(None):
                        container.mux(packet)
                if count != frames:
                    raise ValueError("truncated PCM recording")
                outcome = "not_smaller_passthrough"
                if target.stat().st_size < size:
                    selected = result(target, ".flac", "audio/flac", "lossless_flac")
        except (InterruptedError, AudioLimitError):
            raise
        except Exception as exc:
            # A preparation error must not discard a recording. Never log paths,
            # input contents, decoder messages or credentials.
            outcome = "encoder_error_passthrough"
            LOGGER.warning("Lossless preparation retained original: %s", type(exc).__name__)
        yield selected or result(path, suffix, mime, outcome)


def decode_bounded(path: Path, *, timeout_seconds: float = 20,
                   max_pcm_bytes: int = MAX_AUDIO_BYTES,
                   max_seconds: float = MAX_AUDIO_SECONDS):
    """Decode once at inference, enforcing actual sample limits before allocation.

    The resampling here is the existing Whisper input preprocessing, NOT a
    transport conversion. The uploaded/forwarded source remains unchanged.
    """
    import av
    import numpy as np

    started = time.monotonic()
    sample_bytes = 0
    duration = 0.0
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    output = io.BytesIO()
    try:
        with av.open(str(path), mode="r", metadata_errors="ignore") as container:
            if len(container.streams.audio) != 1:
                raise AudioLimitError("exactly one audio stream is required")
            stream = container.streams.audio[0]
            source_samples = 0

            def bounded_frames():
                nonlocal sample_bytes, duration, source_samples
                for frame in container.decode(audio=0):
                    source_samples += frame.samples
                    sample_bytes += frame.samples * len(frame.layout.channels) * 2
                    duration += frame.samples / frame.sample_rate
                    if (sample_bytes > max_pcm_bytes or duration > max_seconds
                            or time.monotonic() - started > timeout_seconds):
                        raise AudioLimitError("decoded audio exceeds resource limits")
                    yield frame

            fifo = av.AudioFifo()
            for frame in bounded_frames():
                frame.pts = None
                fifo.write(frame)
                if fifo.samples >= 500000:
                    for converted in resampler.resample(fifo.read()):
                        output.write(converted.to_ndarray().tobytes())
            if fifo.samples:
                for converted in resampler.resample(fifo.read()):
                    output.write(converted.to_ndarray().tobytes())
            for converted in resampler.resample(None):
                output.write(converted.to_ndarray().tobytes())
            if stream.codec_context.name == "flac" and stream.duration:
                expected_samples = round(stream.duration * stream.time_base * stream.sample_rate)
                if source_samples != expected_samples:
                    raise AudioLimitError("truncated lossless audio")
        if output.tell() == 0:
            raise AudioLimitError("empty decoded audio")
        return np.frombuffer(output.getbuffer(), dtype=np.int16).astype(np.float32) / 32768.0
    except AudioLimitError:
        raise
    except Exception as exc:
        raise AudioLimitError("audio cannot be decoded") from exc


def self_test() -> dict:
    """Small bit-perfect packaged codec test; no recording, model or network."""
    import av
    import numpy as np
    with tempfile.TemporaryDirectory(prefix="ctrlspeak-codec-test-") as directory:
        wav = Path(directory) / "test.wav"
        pcm = (np.sin(np.arange(44100) * 0.05) * 12000).astype("<i2").tobytes()
        with wave.open(str(wav), "wb") as source:
            source.setnchannels(1)
            source.setsampwidth(2)
            source.setframerate(44100)
            source.writeframes(pcm)
        with prepare_audio(wav) as prepared:
            assert prepared.outcome == "lossless_flac"
            with av.open(str(prepared.path)) as source:
                restored = b"".join(frame.to_ndarray().tobytes() for frame in source.decode(audio=0))
            assert restored == pcm
            return {"protocol": TRANSPORT_PROTOCOL, "bit_perfect": True,
                    "version": RELEASE_VERSION, "codec": "flac"}
