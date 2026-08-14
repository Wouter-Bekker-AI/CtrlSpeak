from __future__ import annotations

import ipaddress
import logging
import os
import re
import secrets
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.corrections import CorrectionStore
from app.languages import choose_allowed_language, normalize_language_policy


SERVICE_VERSION = "0.5.2"
LOGGER = logging.getLogger("ctrlspeak_whisper_transcription")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(os.environ.get("WHISPER_DATA_DIR", ROOT / "data"))
DEFAULT_MAX_UPLOAD_BYTES = int(os.environ.get("WHISPER_MAX_UPLOAD_BYTES", 100 * 1024 * 1024))
DEFAULT_BEARER_TOKEN = os.environ.get("WHISPER_BEARER_TOKEN") or None
MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "large-v3-turbo").strip() or "large-v3-turbo"
MODEL_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda").strip().casefold() or "cuda"
MODEL_COMPUTE_TYPE = (
    os.environ.get(
        "WHISPER_COMPUTE_TYPE",
        "float16" if MODEL_DEVICE == "cuda" else "int8",
    ).strip().casefold()
    or ("float16" if MODEL_DEVICE == "cuda" else "int8")
)
MODEL_CPU_THREADS = max(1, int(os.environ.get("WHISPER_CPU_THREADS", "4")))
MODEL_NUM_WORKERS = max(1, int(os.environ.get("WHISPER_NUM_WORKERS", "1")))


def collect_text_from_segments(segments: Iterable[Mapping[str, Any]]) -> str:
    def normalize_for_comparison(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip().lower())
        cleaned = re.sub(r"[\"'“”‘’]+$", "", cleaned)
        return re.sub(r"[\s.,!?;:]+$", "", cleaned)

    pieces = []
    for segment in segments:
        text = segment.get("text")
        if isinstance(text, str) and (text := text.strip()):
            pieces.append(text)
    while len(pieces) >= 2:
        if normalize_for_comparison(pieces[-1]) != normalize_for_comparison(pieces[-2]):
            break
        pieces.pop()
    return " ".join(pieces)


class Backend(Protocol):
    def load(self) -> None: ...

    def transcribe(
        self,
        audio_path: Path,
        allowed_languages: tuple[str, ...],
        initial_prompt: str | None,
        word_timestamps: bool,
    ) -> dict[str, Any]: ...


class FasterWhisperBackend:
    def __init__(
        self,
        model_dir: Path,
        *,
        model_name: str = MODEL_NAME,
        device: str = MODEL_DEVICE,
        compute_type: str = MODEL_COMPUTE_TYPE,
        cpu_threads: int = MODEL_CPU_THREADS,
        num_workers: int = MODEL_NUM_WORKERS,
    ) -> None:
        self.model_dir = model_dir
        self.name = model_name.strip() or "large-v3-turbo"
        self.device = device.strip().casefold()
        self.compute_type = compute_type.strip().casefold()
        self.cpu_threads = max(1, cpu_threads)
        self.num_workers = max(1, num_workers)
        self.model: Any | None = None

    def load(self) -> None:
        """Load only the explicitly configured runtime; never silently fall back."""
        try:
            import ctranslate2
            from faster_whisper import WhisperModel

            supported = ctranslate2.get_supported_compute_types(self.device)
            if self.compute_type not in supported:
                raise RuntimeError(
                    f"{self.device} {self.compute_type} is unavailable; "
                    f"supported compute types: {sorted(supported)}"
                )
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.model = WhisperModel(
                self.name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.model_dir),
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Whisper {self.name} with {self.device}/{self.compute_type}. "
                "Automatic device or compute-type fallback is intentionally disabled."
            ) from exc

    def _detect_language(self, audio_path: Path) -> str | None:
        if self.model is None:
            raise RuntimeError("model is not loaded")
        try:
            from faster_whisper.audio import decode_audio

            audio = decode_audio(str(audio_path))
            detected, _probability, _all_probabilities = self.model.detect_language(
                audio=audio,
                vad_filter=True,
            )
            return str(detected).strip().casefold() or None
        except Exception:
            LOGGER.warning(
                "Language detection failed before allowlist selection; using the ordered fallback",
                exc_info=True,
            )
            return None

    def transcribe(
        self,
        audio_path: Path,
        allowed_languages: tuple[str, ...],
        initial_prompt: str | None,
        word_timestamps: bool,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("model is not loaded")
        detected_language = self._detect_language(audio_path) if len(allowed_languages) > 1 else None
        selected_language = choose_allowed_language(allowed_languages, detected_language)
        segments, info = self.model.transcribe(
            str(audio_path),
            language=selected_language,
            task="transcribe",
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        result_segments = []
        for segment in segments:
            item: dict[str, Any] = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            if word_timestamps:
                item["words"] = [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                    for word in (segment.words or [])
                ]
            else:
                item["words"] = []
            result_segments.append(item)

        reported_language = str(info.language or "").strip().casefold()
        if allowed_languages and reported_language not in allowed_languages:
            raise RuntimeError(
                "Whisper returned a language outside the requested output-language allowlist"
            )
        return {
            "raw_text": collect_text_from_segments(result_segments),
            "language": reported_language,
            "detected_language": detected_language or reported_language,
            "allowed_languages": list(allowed_languages),
            "segments": result_segments,
        }


class CorrectionCreate(BaseModel):
    source_phrase: str = Field(min_length=1, max_length=1000)
    replacement_phrase: str = Field(min_length=1, max_length=1000)
    context_terms: list[str] = Field(default_factory=list, max_length=50)
    tags: list[str] = Field(default_factory=list, max_length=50)
    enabled: bool = True
    priority: int = Field(default=0, ge=-1000, le=1000)


class CorrectionUpdate(BaseModel):
    source_phrase: str | None = Field(default=None, min_length=1, max_length=1000)
    replacement_phrase: str | None = Field(default=None, min_length=1, max_length=1000)
    context_terms: list[str] | None = Field(default=None, max_length=50)
    tags: list[str] | None = Field(default=None, max_length=50)
    enabled: bool | None = None
    priority: int | None = Field(default=None, ge=-1000, le=1000)


class FeedbackRequest(BaseModel):
    rule_ids: list[str] = Field(default_factory=list, max_length=100)
    note: str | None = Field(default=None, max_length=2000)
    confirmed_text: str | None = Field(default=None, min_length=1, max_length=100_000)
    capture_method: str | None = Field(default=None, max_length=100)
    client_metadata: dict[str, Any] = Field(default_factory=dict)


def _is_loopback_client(host: str | None) -> bool:
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)


def create_app(
    *,
    store: CorrectionStore | None = None,
    backend: Backend | None = None,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    temp_dir: Path | None = None,
    bearer_token: str | None = DEFAULT_BEARER_TOKEN,
) -> FastAPI:
    data_dir = DEFAULT_DATA_DIR
    store = store or CorrectionStore(data_dir / "corrections.sqlite3")
    backend = backend or FasterWhisperBackend(data_dir / "models")
    temp_dir = temp_dir or data_dir / "uploads"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        try:
            await run_in_threadpool(backend.load)
            app.state.ready = True
            LOGGER.info(
                "Whisper model ready: %s on %s",
                getattr(backend, "name", "fake"),
                getattr(backend, "device", "fake"),
            )
            yield
        except Exception:
            LOGGER.exception("Whisper startup failed; refusing automatic runtime fallback")
            raise
        finally:
            app.state.ready = False

    app = FastAPI(
        title="CtrlSpeak Whisper Transcription API",
        version=SERVICE_VERSION,
        description=(
            "Configured Whisper backend for CtrlSpeak. The multipart allowed_languages field "
            "is an ordered, server-enforced output-language allowlist."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def require_lan_bearer_auth(request: Request, call_next):
        client_host = request.client.host if request.client else None
        if _is_loopback_client(client_host):
            return await call_next(request)
        if not bearer_token:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "non-loopback access is disabled until WHISPER_BEARER_TOKEN is configured"
                },
            )
        authorization = request.headers.get("authorization", "")
        scheme, separator, supplied = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not secrets.compare_digest(supplied, bearer_token)
        ):
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content={"detail": "valid bearer authentication is required for non-loopback access"},
            )
        return await call_next(request)

    @app.get("/health", tags=["service"])
    async def health(request: Request) -> dict[str, str]:
        if not request.app.state.ready:
            raise HTTPException(503, "model is not ready")
        return {
            "status": "ready",
            "version": SERVICE_VERSION,
            "model": getattr(backend, "name", "fake"),
            "device": getattr(backend, "device", "fake"),
            "compute_type": getattr(backend, "compute_type", "fake"),
        }

    @app.post("/v1/corrections", status_code=201, tags=["corrections"])
    async def create_correction(payload: CorrectionCreate) -> dict[str, Any]:
        try:
            return store.create_rule(**payload.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/corrections", tags=["corrections"])
    async def list_corrections(enabled: bool | None = None) -> dict[str, list[dict[str, Any]]]:
        return {"items": store.list_rules(enabled)}

    @app.get("/v1/corrections/{rule_id}", tags=["corrections"])
    async def get_correction(rule_id: str) -> dict[str, Any]:
        rule = store.get_rule(rule_id)
        if not rule:
            raise HTTPException(404, "correction rule not found")
        return rule

    @app.patch("/v1/corrections/{rule_id}", tags=["corrections"])
    async def update_correction(rule_id: str, payload: CorrectionUpdate) -> dict[str, Any]:
        try:
            rule = store.update_rule(rule_id, payload.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not rule:
            raise HTTPException(404, "correction rule not found")
        return rule

    @app.delete("/v1/corrections/{rule_id}", status_code=204, tags=["corrections"])
    async def delete_correction(rule_id: str) -> Response:
        if not store.delete_rule(rule_id):
            raise HTTPException(404, "correction rule not found")
        return Response(status_code=204)

    @app.post(
        "/v1/transcribe",
        tags=["transcription"],
        summary="Transcribe audio with an optional server-enforced language allowlist",
    )
    async def transcribe(
        audio: UploadFile = File(..., description="WAV or other faster-whisper-compatible audio"),
        allowed_languages: str | None = Form(
            None,
            description=(
                "Ordered comma-separated Whisper language codes, for example 'en' or 'en,af'. "
                "The response language cannot be outside this list; the first code is the fallback. "
                "Omit for automatic language detection. Maximum five."
            ),
        ),
        language: str | None = Form(
            None,
            description="Legacy single-language alias. Prefer allowed_languages.",
        ),
        initial_prompt: str | None = Form(None),
        context: str | None = Form(None),
        word_timestamps: bool = Form(False),
    ) -> dict[str, Any]:
        if not app.state.ready:
            raise HTTPException(503, "model is not ready")
        try:
            language_policy = normalize_language_policy(allowed_languages, language)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        temp_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(audio.filename or "upload").suffix[:16]
        temp_path: Path | None = None
        total = 0
        try:
            with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=suffix, delete=False) as uploaded:
                temp_path = Path(uploaded.name)
                while chunk := await audio.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_upload_bytes:
                        raise HTTPException(413, f"audio exceeds {max_upload_bytes} byte upload limit")
                    uploaded.write(chunk)
            result = await run_in_threadpool(
                backend.transcribe,
                temp_path,
                language_policy,
                initial_prompt,
                word_timestamps,
            )
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception("transcription failed")
            raise HTTPException(500, "transcription failed") from exc
        finally:
            await audio.close()
            if temp_path:
                temp_path.unlink(missing_ok=True)

        result_language = str(result.get("language") or "").strip().casefold()
        if language_policy and result_language not in language_policy:
            LOGGER.error(
                "Blocking response outside requested language policy: reported=%r allowed=%r",
                result_language,
                language_policy,
            )
            raise HTTPException(500, "transcription violated the requested language policy")

        raw_text = result["raw_text"]
        corrected_text, applied_ids, exact_override_id = store.apply_with_metadata(raw_text, context)
        transcription_id = store.create_transcription(
            raw_text=raw_text,
            corrected_text=corrected_text,
            language=result_language,
            segments=result["segments"],
            applied_rule_ids=applied_ids,
            context=context,
            metadata={
                "word_timestamps": word_timestamps,
                "initial_prompt_supplied": bool(initial_prompt),
                "original_filename": audio.filename,
                "content_type": audio.content_type,
                "allowed_languages": list(language_policy),
                "detected_language": result.get("detected_language"),
            },
        )
        return {
            "id": transcription_id,
            "raw_text": raw_text,
            "text": corrected_text,
            "language": result_language,
            "detected_language": result.get("detected_language"),
            "allowed_languages": list(language_policy),
            "language_policy": "restricted" if language_policy else "automatic",
            "segments": result["segments"],
            "applied_correction_rule_ids": applied_ids,
            "exact_override_id": exact_override_id,
        }

    @app.post("/v1/transcriptions/{transcription_id}/feedback", tags=["transcription"])
    async def feedback(transcription_id: str, payload: FeedbackRequest) -> dict[str, Any]:
        if not payload.rule_ids and payload.confirmed_text is None:
            raise HTTPException(422, "provide rule_ids and/or confirmed_text")
        attached: list[str] = []
        if payload.rule_ids:
            try:
                result = store.attach_feedback(transcription_id, payload.rule_ids, payload.note)
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
            if result is None:
                raise HTTPException(404, "transcription not found")
            attached = result

        approval: dict[str, str] = {}
        if payload.confirmed_text is not None:
            if len(payload.client_metadata) > 100:
                raise HTTPException(422, "client_metadata has too many entries")
            try:
                approved = store.approve_exact_override(
                    transcription_id,
                    confirmed_text=payload.confirmed_text,
                    capture_method=payload.capture_method or "unspecified",
                    client_metadata=payload.client_metadata,
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if approved is None:
                raise HTTPException(404, "transcription not found")
            approval = approved
        return {
            "transcription_id": transcription_id,
            "rule_ids": attached,
            "note": payload.note,
            **approval,
        }

    return app


app = create_app()
