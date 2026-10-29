from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.auth import Authenticator, Principal
from app.admission import Admission, RequestTelemetryMiddleware, WorkOwner, milliseconds, safe_timings
from app.audio_transport import AudioLimitError, capabilities as audio_capabilities, decode_bounded
from app.corrections import CorrectionStore
from app.languages import choose_allowed_language, normalize_language_policy
from app.normalization import (
    MODEL_ATTRIBUTION, S1MiniNormalizer, build_worker_normalizer,
    metadata as cleanup_metadata, safe_worker_result,
)
from app.providers import (
    LocalWhisperProvider,
    OpenAITranscriptionProvider,
    ProviderContext,
    ProviderFailure,
    ProviderRouter,
    RemoteWorkerProvider,
)


SERVICE_VERSION = "0.8.1"
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
SERVICE_ROLES = {"standalone", "gateway", "worker"}


def collect_text_from_segments(segments: Iterable[Mapping[str, Any]]) -> str:
    def normalize_for_comparison(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip().lower())
        cleaned = re.sub(r'["\'“”‘’]+$', "", cleaned)
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

    def _detect_language(self, audio) -> str | None:
        if self.model is None:
            raise RuntimeError("model is not loaded")
        try:
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
        audio = decode_bounded(audio_path)
        detected_language = self._detect_language(audio) if len(allowed_languages) > 1 else None
        selected_language = choose_allowed_language(allowed_languages, detected_language)
        segments, info = self.model.transcribe(
            audio,
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
            "detected_languages": [detected_language or reported_language]
            if (detected_language or reported_language)
            else [],
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
    scope: str = Field(default="user", pattern="^(user|global)$")
    language_codes: list[str] = Field(default_factory=list, max_length=5)
    send_as_keyword: bool = False


class CorrectionUpdate(BaseModel):
    source_phrase: str | None = Field(default=None, min_length=1, max_length=1000)
    replacement_phrase: str | None = Field(default=None, min_length=1, max_length=1000)
    context_terms: list[str] | None = Field(default=None, max_length=50)
    tags: list[str] | None = Field(default=None, max_length=50)
    enabled: bool | None = None
    priority: int | None = Field(default=None, ge=-1000, le=1000)
    language_codes: list[str] | None = Field(default=None, max_length=5)
    send_as_keyword: bool | None = None


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


def _service_role(value: str | None) -> str:
    role = (value or "standalone").strip().casefold()
    if role not in SERVICE_ROLES:
        raise ValueError(f"CTRLSPEAK_SERVICE_ROLE must be one of {sorted(SERVICE_ROLES)}")
    return role


def _build_default_backend(role: str, data_dir: Path) -> FasterWhisperBackend:
    if role != "gateway":
        return FasterWhisperBackend(data_dir / "models")
    return FasterWhisperBackend(
        data_dir / "models",
        model_name=os.environ.get("CTRLSPEAK_FALLBACK_MODEL", "tiny").strip() or "tiny",
        device="cpu",
        compute_type=os.environ.get("CTRLSPEAK_FALLBACK_COMPUTE_TYPE", "int8").strip() or "int8",
        cpu_threads=max(1, int(os.environ.get("CTRLSPEAK_FALLBACK_CPU_THREADS", "2"))),
        num_workers=1,
    )


def _build_router(role: str, backend: Backend, environ: Mapping[str, str]) -> ProviderRouter:
    if role == "standalone":
        local = LocalWhisperProvider("local-whisper", backend)
        return ProviderRouter([local], {"local": (local.id,)}, default_strategy="local")
    if role != "gateway":
        raise ValueError("a provider router is only valid for gateway or standalone roles")

    providers: list[Any] = []
    worker: RemoteWorkerProvider | None = None
    worker_url = (environ.get("CTRLSPEAK_WORKER_URL") or "").strip()
    worker_token = (environ.get("CTRLSPEAK_WORKER_TOKEN") or "").strip()
    if bool(worker_url) != bool(worker_token):
        raise ValueError("CTRLSPEAK_WORKER_URL and CTRLSPEAK_WORKER_TOKEN must be configured together")
    if worker_url:
        worker = RemoteWorkerProvider(
            "ubuntu-gpu-large-v3-turbo",
            worker_url,
            worker_token,
            timeout_seconds=float(environ.get("CTRLSPEAK_PROVIDER_TIMEOUT_SECONDS", "90")),
            connect_timeout_seconds=float(
                environ.get("CTRLSPEAK_WORKER_CONNECT_TIMEOUT_SECONDS", "0.35")
            ),
            health_probe_timeout_seconds=float(
                environ.get("CTRLSPEAK_WORKER_HEALTH_TIMEOUT_SECONDS", "0.5")
            ),
            health_cache_seconds=float(
                environ.get("CTRLSPEAK_WORKER_HEALTH_CACHE_SECONDS", "5")
            ),
            circuit_break_seconds=float(
                environ.get("CTRLSPEAK_WORKER_CIRCUIT_BREAK_SECONDS", "30")
            ),
        )
        providers.append(worker)

    openai = OpenAITranscriptionProvider(
        model=(environ.get("CTRLSPEAK_OPENAI_TRANSCRIBE_MODEL") or "gpt-transcribe").strip(),
        timeout_seconds=float(environ.get("CTRLSPEAK_PROVIDER_TIMEOUT_SECONDS", "90")),
    )
    fallback = LocalWhisperProvider("nova-tiny-whisper", backend, lazy=True)
    providers.extend([openai, fallback])
    worker_ids = [worker.id] if worker is not None else []
    ubuntu_first = tuple([*worker_ids, openai.id, fallback.id])
    openai_first = tuple([openai.id, *worker_ids, fallback.id])
    private_first = tuple([*worker_ids, fallback.id])
    strategies: dict[str, tuple[str, ...]] = {
        "ubuntu-gpu-preferred": ubuntu_first,
        "openai-preferred": openai_first,
        "openai-only": (openai.id,),
        "gateway-tiny-only": (fallback.id,),
        # Compatibility aliases retained for existing v0.6.0/v0.6.1 clients.
        "resilient-quality": ubuntu_first,
        "private-first": private_first,
        "openai-then-local": (openai.id, fallback.id),
        "local-fallback-only": (fallback.id,),
    }
    if worker is not None:
        strategies["ubuntu-gpu-only"] = (worker.id,)
    return ProviderRouter(
        providers,
        strategies,
        default_strategy=(
            environ.get("CTRLSPEAK_DEFAULT_STRATEGY") or "ubuntu-gpu-preferred"
        ).strip(),
    )


def _principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if not isinstance(principal, Principal):
        raise HTTPException(401, "authenticated client identity is required")
    return principal


def _allowed_provider_ids(principal: Principal, router: ProviderRouter) -> set[str]:
    if "*" in principal.allowed_providers:
        return set(router.providers)
    return {item for item in principal.allowed_providers if item in router.providers}


def _normalize_rule_languages(codes: list[str]) -> list[str]:
    try:
        return list(normalize_language_policy(",".join(codes) if codes else None, None))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def create_app(
    *,
    store: CorrectionStore | None = None,
    backend: Backend | None = None,
    router: ProviderRouter | None = None,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    temp_dir: Path | None = None,
    bearer_token: str | None = DEFAULT_BEARER_TOKEN,
    clients_json: str | None = None,
    worker_token: str | None = None,
    service_role: str | None = None,
    environ: Mapping[str, str] | None = None,
    normalizer: S1MiniNormalizer | None = None,
) -> FastAPI:
    environ = os.environ if environ is None else environ
    role = _service_role(service_role or environ.get("CTRLSPEAK_SERVICE_ROLE"))
    data_dir = Path(environ.get("WHISPER_DATA_DIR", str(DEFAULT_DATA_DIR)))
    store = None if role == "worker" else (store or CorrectionStore(data_dir / "corrections.sqlite3"))
    backend = backend or _build_default_backend(role, data_dir)
    # Even stale gateway CPU-normalization configuration cannot enable cleanup here.
    normalizer = (normalizer or build_worker_normalizer(
        environ, role=role, device=getattr(backend, "device", "unknown"),
    )) if role == "worker" and getattr(backend, "device", "unknown") == "cuda" else None
    temp_dir = temp_dir or data_dir / "uploads"
    clients_json = clients_json if clients_json is not None else environ.get("CTRLSPEAK_CLIENTS_JSON")
    worker_token = worker_token if worker_token is not None else environ.get("CTRLSPEAK_WORKER_TOKEN")
    authenticator = Authenticator(
        clients_json=clients_json,
        legacy_token=bearer_token,
        worker_token=worker_token,
    )
    request_deadline = float(environ.get("CTRLSPEAK_REQUEST_DEADLINE_SECONDS", "75" if role == "worker" else "150"))
    if not 0 < request_deadline <= 600:
        raise ValueError("CTRLSPEAK_REQUEST_DEADLINE_SECONDS must be between 0 and 600")
    capacity = int(environ.get("CTRLSPEAK_MAX_ACTIVE_REQUESTS", "1" if role == "worker" else "8"))
    if not 1 <= capacity <= 32:
        raise ValueError("CTRLSPEAK_MAX_ACTIVE_REQUESTS must be between 1 and 32")
    admission = Admission(capacity, request_deadline)
    owned_tasks: set[asyncio.Task] = set()
    if role != "worker":
        router = router or _build_router(role, backend, environ)
    eager_provider = None
    if role == "worker":
        eager_provider = LocalWhisperProvider("ubuntu-gpu-large-v3-turbo", backend)
    elif role == "standalone" and router:
        eager_provider = router.providers["local-whisper"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        try:
            if eager_provider is not None:
                await run_in_threadpool(eager_provider.load)
            app.state.ready = True
            LOGGER.info("CtrlSpeak service ready: role=%s version=%s", role, SERVICE_VERSION)
            yield
        except Exception:
            LOGGER.exception("CtrlSpeak service startup failed")
            raise
        finally:
            app.state.ready = False
            if normalizer is not None:
                await normalizer.close()

    app = FastAPI(
        title="CtrlSpeak Transcription API",
        version=SERVICE_VERSION,
        description=(
            "Role-aware CtrlSpeak transcription service. Gateways route requests and apply "
            "identity-scoped corrections; workers return raw output and optional GPU cleanup."
        ),
        lifespan=lifespan,
    )
    app.state.role = role
    app.state.router = router
    app.state.admission = admission
    app.state.owned_tasks = owned_tasks

    async def admitted_call(request: Request, call_next):
        if request.url.path not in {"/v1/transcribe", "/v1/worker/transcribe"}:
            return await call_next(request)
        lease = admission.acquire()
        if lease is None:
            category = "worker_busy" if role == "worker" else "gateway_busy"
            request.state.failure_category = category
            return JSONResponse(status_code=503, headers={"Retry-After": "1"}, content={
                "detail": {"message": "Transcription capacity is busy; retry shortly.",
                           "category": category, "retryable": True,
                           "request_id": request.state.request_id,
                           "admission": admission.snapshot()},
            })
        request.state.lease = lease
        try:
            return await asyncio.wait_for(call_next(request), timeout=request_deadline + 0.1)
        except asyncio.TimeoutError:
            request.state.cancel_event.set()
            category = "worker_deadline_exceeded" if role == "worker" else "request_deadline_exceeded"
            request.state.failure_category = category
            return JSONResponse(status_code=504, content={"detail": {
                "message": "Transcription exceeded its request deadline.",
                "category": category, "retryable": True,
                "request_id": request.state.request_id,
            }})
        except asyncio.CancelledError:
            request.state.cancel_event.set()
            raise
        finally:
            # Ownership is moved to the actual job before inference starts. A
            # disconnected caller must never make still-running work look idle.
            if not lease.transferred:
                lease.release()

    async def owned_job(request: Request, temp_path: Path, operation):
        # Reject before scheduling: a body that exhausted the request budget
        # must not start even a millisecond of new inference or paid fallback.
        remaining = request_deadline - (time.monotonic() - request.state.request_started_at)
        if remaining <= 0 or request.state.cancel_event.is_set():
            temp_path.unlink(missing_ok=True)
            category = ("request_cancelled" if request.state.cancel_event.is_set() else
                        "worker_deadline_exceeded" if role == "worker" else "request_deadline_exceeded")
            request.state.failure_category = category
            raise HTTPException(499 if category == "request_cancelled" else 504, {
                "message": "Request ended before transcription could begin.",
                "category": category, "retryable": category != "request_cancelled",
                "request_id": request.state.request_id,
            })
        lease = request.state.lease
        lease.transferred = True
        owner = WorkOwner(lease, temp_path)
        request.state.work_owner = owner

        async def run():
            try:
                return await operation()
            finally:
                owner.close()

        task = asyncio.create_task(run())
        owned_tasks.add(task)

        def finished(completed):
            owned_tasks.discard(completed)
            if not completed.cancelled():
                completed.exception()  # Retrieve errors even after caller disconnect.

        task.add_done_callback(finished)

        async def disconnected():
            # Multipart parsing has consumed the complete body before owned_job
            # is invoked. This is now the sole ASGI receive consumer; it must be
            # stopped before the endpoint returns and response streaming begins.
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    request.state.cancel_event.set()
                    return

        disconnect_task = asyncio.create_task(disconnected())
        try:
            # Shielding is intentional: Python cannot kill a running inference
            # thread. Its permit and input remain owned until it really exits.
            done, _ = await asyncio.wait({task, disconnect_task}, timeout=remaining,
                                         return_when=asyncio.FIRST_COMPLETED)
            if disconnect_task in done:
                request.state.failure_category = "request_cancelled"
                raise HTTPException(499, {"message": "Caller disconnected.",
                                         "category": "request_cancelled", "retryable": False,
                                         "request_id": request.state.request_id})
            if task in done:
                return task.result()
            request.state.cancel_event.set()
            category = "worker_deadline_exceeded" if role == "worker" else "request_deadline_exceeded"
            request.state.failure_category = category
            raise HTTPException(504, {"message": "Transcription exceeded its request deadline.",
                                     "category": category, "retryable": True,
                                     "request_id": request.state.request_id})
        except asyncio.CancelledError:
            request.state.cancel_event.set()
            raise
        finally:
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)

    def resource_health() -> dict[str, Any]:
        # Root/system drive health: inspection only, never automatic deletion.
        try:
            disk = shutil.disk_usage(Path.cwd().anchor or "/")
            used_percent = round((disk.total - disk.free) / disk.total * 100, 2)
            return {"disk_free_bytes": disk.free, "disk_used_percent": used_percent,
                    "disk_warning": disk.free < 5 * 1024**3 or used_percent > 90}
        except OSError:
            return {"disk_warning": None}

    @app.middleware("http")
    async def require_bearer_auth(request: Request, call_next):
        client_host = request.client.host if request.client else None
        if _is_loopback_client(client_host):
            request.state.principal = Principal("legacy", admin=True)
            request.state.worker_authenticated = True
            return await admitted_call(request, call_next)

        authorization = request.headers.get("authorization")
        if role == "worker":
            if not authenticator.has_worker_credential:
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "worker access is disabled until CTRLSPEAK_WORKER_TOKEN is configured"
                    },
                )
            if not authenticator.authenticate_worker(authorization):
                return JSONResponse(
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                    content={"detail": "valid worker bearer authentication is required"},
                )
            request.state.principal = Principal("gateway-service")
            request.state.worker_authenticated = True
            return await admitted_call(request, call_next)

        if not authenticator.has_client_credentials:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "non-loopback access is disabled until client credentials are configured"
                },
            )
        principal = authenticator.authenticate_client(authorization)
        if principal is None:
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content={"detail": "valid client bearer authentication is required"},
            )
        request.state.principal = principal
        return await admitted_call(request, call_next)

    # Outer ASGI wrapper measures body receipt (including multipart parsing),
    # generates a bounded correlation ID and never records bodies or credentials.
    app.add_middleware(RequestTelemetryMiddleware)

    @app.get("/health", tags=["service"])
    async def health(request: Request) -> dict[str, Any]:
        if not request.app.state.ready:
            raise HTTPException(503, "service is not ready")
        payload: dict[str, Any] = {
            "status": "ready",
            "version": SERVICE_VERSION,
            "role": role,
            "revision": os.environ.get("CTRLSPEAK_RELEASE_REVISION", "source"),
            "audio_transport": audio_capabilities(),
            "admission": admission.snapshot(),
            "resources": resource_health(),
        }
        if role in {"worker", "standalone"}:
            payload.update(
                {
                    "model": getattr(backend, "name", "unknown"),
                    "device": getattr(backend, "device", "unknown"),
                    "compute_type": getattr(backend, "compute_type", "unknown"),
                }
            )
        return payload

    @app.get("/v1/capabilities", tags=["service"])
    async def capabilities(request: Request) -> dict[str, Any]:
        if not request.app.state.ready:
            raise HTTPException(503, "service is not ready")
        if role == "worker":
            return {
                "version": SERVICE_VERSION,
                "role": role,
                "revision": os.environ.get("CTRLSPEAK_RELEASE_REVISION", "source"),
                "audio_transport": audio_capabilities(),
                "admission": admission.snapshot(),
                "request_deadline_seconds": request_deadline,
                "accepts_client_transcriptions": False,
                "applies_corrections": False,
                "text_normalization": {
                    "enabled": normalizer is not None, "model": MODEL_ATTRIBUTION,
                    "request_field": "cleanup", "default": False, "device": "cuda",
                    "location": "ubuntu-worker", "language": "en", "fail_open": True,
                },
                "telemetry": {
                    "inference_duration_ms": True,
                    "request_id": True, "stage_timings": True,
                },
                "provider": {
                    "id": "ubuntu-gpu-large-v3-turbo",
                    "kind": "local_whisper",
                    "model": getattr(backend, "name", "unknown"),
                    "device": getattr(backend, "device", "unknown"),
                    "compute_type": getattr(backend, "compute_type", "unknown"),
                    "status": admission.snapshot()["status"],
                },
            }
        principal = _principal(request)
        assert router is not None
        description = await run_in_threadpool(
            router.describe,
            _allowed_provider_ids(principal, router),
        )
        return {
            "version": SERVICE_VERSION,
            "role": role,
            "revision": os.environ.get("CTRLSPEAK_RELEASE_REVISION", "source"),
            "audio_transport": audio_capabilities(),
            "admission": admission.snapshot(),
            "request_deadline_seconds": request_deadline,
            "accepts_client_transcriptions": True,
            "applies_corrections": True,
            "openai_key_storage": "request_only",
            "text_normalization": {
                "mode": "optional-worker-gpu", "model": MODEL_ATTRIBUTION,
                "request_field": "cleanup", "default": False, "language": "en",
                "eligible_providers": ["ubuntu-gpu-large-v3-turbo"],
                "location": "ubuntu-worker", "device": "cuda", "fail_open": True,
                "gateway_cpu_cleanup": False,
            },
            "telemetry": {
                "attempt_duration_ms": True,
                "routing_duration_ms": True,
                "worker_inference_duration_ms": True,
                "worker_health": True,
                "request_id": True,
                "stage_timings": True,
            },
            **description,
        }

    async def receive_audio(audio: UploadFile) -> Path:
        temp_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(audio.filename or "upload").suffix[:16]
        total = 0
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=suffix, delete=False) as uploaded:
                temp_path = Path(uploaded.name)
                while chunk := await audio.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_upload_bytes:
                        raise HTTPException(413, f"audio exceeds {max_upload_bytes} byte upload limit")
                    uploaded.write(chunk)
            return temp_path
        except BaseException:
            if temp_path:
                temp_path.unlink(missing_ok=True)
            raise

    if role == "worker":

        @app.post("/v1/worker/transcribe", tags=["worker"])
        async def worker_transcribe(
            request: Request,
            audio: UploadFile = File(...),
            allowed_languages: str | None = Form(None),
            language: str | None = Form(None),
            initial_prompt: str | None = Form(None),
            keywords: str | None = Form(None),
            word_timestamps: bool = Form(False),
            cleanup: bool = Form(False),
        ) -> dict[str, Any]:
            if not app.state.ready:
                raise HTTPException(503, "worker model is not ready")
            try:
                language_policy = normalize_language_policy(allowed_languages, language)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            try:
                decoded_keywords = json.loads(keywords) if keywords else []
            except json.JSONDecodeError as exc:
                raise HTTPException(422, "keywords must be a JSON string array") from exc
            if not isinstance(decoded_keywords, list) or not all(
                isinstance(item, str) for item in decoded_keywords
            ):
                raise HTTPException(422, "keywords must be a JSON string array")
            decoded_keywords = [item.strip() for item in decoded_keywords if item.strip()][:100]
            effective_prompt = initial_prompt
            if decoded_keywords:
                vocabulary = "Known spellings: " + ", ".join(decoded_keywords)
                effective_prompt = f"{initial_prompt}\n{vocabulary}" if initial_prompt else vocabulary
            spool_started = time.monotonic()
            try:
                temp_path = await receive_audio(audio)
            finally:
                await audio.close()
            spool_duration = milliseconds(spool_started)

            async def process():
                inference_started_at = time.monotonic()
                def infer():
                    # Re-check in the actual inference thread, not just before
                    # scheduling it, so cancellation while waiting for a thread
                    # does not start new GPU work.
                    if request.state.cancel_event.is_set():
                        raise ProviderFailure("Caller cancelled the request", category="request_cancelled",
                                              retryable=False, status_code=499)
                    if time.monotonic() - request.state.request_started_at >= request_deadline:
                        raise ProviderFailure("Worker request deadline exceeded", category="worker_deadline_exceeded",
                                              retryable=True, status_code=504)
                    return backend.transcribe(temp_path, language_policy, effective_prompt, word_timestamps)
                try:
                    result = await run_in_threadpool(
                        request.state.work_owner.call, infer,
                    )
                except ProviderFailure as exc:
                    request.state.failure_category = exc.category
                    raise HTTPException(exc.status_code, {"message": str(exc), "category": exc.category,
                                                          "retryable": exc.retryable,
                                                          "request_id": request.state.request_id}) from exc
                except AudioLimitError as exc:
                    request.state.failure_category = "invalid_audio"
                    raise HTTPException(422, {"message": "Audio is invalid or exceeds decoding limits",
                                             "category": "invalid_audio", "retryable": False,
                                             "request_id": request.state.request_id}) from exc
                except Exception as exc:
                    request.state.failure_category = "worker_error"
                    LOGGER.warning("worker transcription failed request_id=%s", request.state.request_id)
                    raise HTTPException(500, {"message": "Worker transcription failed",
                                             "category": "worker_error", "retryable": True,
                                             "request_id": request.state.request_id}) from exc
                inference_duration = milliseconds(inference_started_at)
                result["provider"] = "ubuntu-gpu-large-v3-turbo"
                result["inference_duration_ms"] = inference_duration
                result["corrected"] = False
                result["normalized_text"] = None
                result["normalization"] = cleanup_metadata(cleanup, "unavailable" if cleanup else "disabled")
                cleanup_started = time.monotonic()
                if cleanup and normalizer is not None and not request.state.cancel_event.is_set():
                    try:
                        cleaned = await normalizer.normalize(
                            str(result.get("raw_text") or ""),
                            language=str(result.get("language") or "").casefold(),
                        )
                        result["normalized_text"] = cleaned.text
                        result["normalization"] = cleaned.info
                    except Exception:
                        LOGGER.warning("worker cleanup failed request_id=%s", request.state.request_id)
                        result["normalization"] = cleanup_metadata(True, "runtime_error")
                result["request_id"] = request.state.request_id
                result["timings"] = {
                    "receive_duration_ms": request.state.receive_duration_ms,
                    "upload_spool_duration_ms": spool_duration,
                    "queue_duration_ms": 0.0,
                    "inference_duration_ms": inference_duration,
                    "cleanup_duration_ms": milliseconds(cleanup_started),
                    "server_duration_ms": milliseconds(request.state.request_started_at),
                }
                return result

            return await owned_job(request, temp_path, process)

    else:
        assert router is not None
        assert store is not None

        @app.post("/v1/corrections", status_code=201, tags=["corrections"])
        async def create_correction(request: Request, payload: CorrectionCreate) -> dict[str, Any]:
            principal = _principal(request)
            if payload.scope == "global" and not principal.admin:
                raise HTTPException(403, "only an administrator may create global corrections")
            try:
                values = payload.model_dump(exclude={"language_codes"})
                return store.create_rule(
                    **values,
                    owner_id=principal.id if payload.scope == "user" else None,
                    language_codes=_normalize_rule_languages(payload.language_codes),
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

        @app.get("/v1/corrections", tags=["corrections"])
        async def list_corrections(
            request: Request,
            enabled: bool | None = None,
            include_all: bool = Query(False),
        ) -> dict[str, list[dict[str, Any]]]:
            principal = _principal(request)
            if include_all and not principal.admin:
                raise HTTPException(403, "only an administrator may list all correction scopes")
            return {
                "items": store.list_rules(
                    enabled,
                    principal_id=principal.id,
                    include_all=include_all,
                )
            }

        @app.get("/v1/corrections/{rule_id}", tags=["corrections"])
        async def get_correction(request: Request, rule_id: str) -> dict[str, Any]:
            principal = _principal(request)
            rule = store.get_rule(
                rule_id,
                principal_id=principal.id,
                include_all=principal.admin,
            )
            if not rule:
                raise HTTPException(404, "correction rule not found")
            return rule

        @app.patch("/v1/corrections/{rule_id}", tags=["corrections"])
        async def update_correction(
            request: Request,
            rule_id: str,
            payload: CorrectionUpdate,
        ) -> dict[str, Any]:
            principal = _principal(request)
            changes = payload.model_dump(exclude_unset=True)
            if "language_codes" in changes:
                changes["language_codes"] = _normalize_rule_languages(changes["language_codes"])
            try:
                rule = store.update_rule(
                    rule_id,
                    changes,
                    principal_id=principal.id,
                    include_all=principal.admin,
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if not rule:
                raise HTTPException(404, "correction rule not found")
            return rule

        @app.delete("/v1/corrections/{rule_id}", status_code=204, tags=["corrections"])
        async def delete_correction(request: Request, rule_id: str) -> Response:
            principal = _principal(request)
            if not store.delete_rule(
                rule_id,
                principal_id=principal.id,
                include_all=principal.admin,
            ):
                raise HTTPException(404, "correction rule not found")
            return Response(status_code=204)

        @app.post(
            "/v1/transcribe",
            tags=["transcription"],
            summary="Route audio through CtrlSpeak with a server-enforced language allowlist",
        )
        async def transcribe(
            request: Request,
            audio: UploadFile = File(...),
            allowed_languages: str | None = Form(None),
            language: str | None = Form(None),
            initial_prompt: str | None = Form(None),
            context: str | None = Form(None),
            word_timestamps: bool = Form(False),
            strategy: str | None = Form(None),
            provider: str | None = Form(None),
            cleanup: bool = Form(False),
        ) -> dict[str, Any]:
            if not app.state.ready:
                raise HTTPException(503, "service is not ready")
            principal = _principal(request)
            try:
                language_policy = normalize_language_policy(allowed_languages, language)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            rule_language = language_policy[0] if len(language_policy) == 1 else None
            keywords = tuple(
                store.keyword_hints(
                    principal_id=principal.id,
                    language=rule_language,
                )
            )
            provider_context = ProviderContext(
                allowed_languages=language_policy,
                initial_prompt=initial_prompt,
                keywords=keywords,
                word_timestamps=word_timestamps,
                openai_api_key=request.headers.get("x-ctrlspeak-openai-key") or None,
                cleanup=cleanup,
                request_id=request.state.request_id,
                deadline_at=request.state.request_started_at + request_deadline,
                cancel_event=request.state.cancel_event,
            )
            spool_started = time.monotonic()
            try:
                temp_path = await receive_audio(audio)
            finally:
                await audio.close()
            spool_duration = milliseconds(spool_started)

            async def route():
                return await run_in_threadpool(
                    request.state.work_owner.call, router.transcribe,
                    temp_path,
                    provider_context,
                    strategy=strategy,
                    provider_id=provider,
                    allowed_provider_ids=_allowed_provider_ids(principal, router),
                )

            try:
                routed = await owned_job(request, temp_path, route)
            except ProviderFailure as exc:
                request.state.failure_category = exc.category
                detail: dict[str, Any] = {
                    "message": str(exc),
                    "category": exc.category,
                    "retryable": exc.retryable,
                    "attempts": getattr(exc, "attempts", []),
                    "request_id": request.state.request_id,
                    "timings": {"receive_duration_ms": request.state.receive_duration_ms,
                                "upload_spool_duration_ms": spool_duration,
                                "server_duration_ms": milliseconds(request.state.request_started_at)},
                }
                routing_duration_ms = getattr(exc, "routing_duration_ms", None)
                if routing_duration_ms is not None:
                    detail["routing_duration_ms"] = routing_duration_ms
                if exc.action:
                    detail["action"] = exc.action
                raise HTTPException(exc.status_code, detail) from exc

            result = routed.result
            result_language = str(result.get("language") or "").strip().casefold()
            detected_languages = [
                str(item).casefold() for item in result.get("detected_languages", []) if item
            ]
            if language_policy and (
                (result_language and result_language not in language_policy)
                or any(item not in language_policy for item in detected_languages)
            ):
                raise HTTPException(502, "transcription violated the requested language policy")
            raw_text = str(result.get("raw_text") or "")
            corrected_text, applied_ids, exact_override_id = store.apply_with_metadata(
                raw_text,
                context,
                principal_id=principal.id,
                language=result_language or rule_language,
            )
            normalized_text, normalization = safe_worker_result(
                result, requested=cleanup, provider=routed.provider_id, language=result_language,
            )
            s1_cleaned_text = normalized_text
            normalized_ids = []
            if exact_override_id is not None:
                normalized_text = None
                normalization = cleanup_metadata(cleanup, "exact_override" if cleanup else "disabled")
            elif normalized_text is not None:
                normalized_text, normalized_ids = store.preview_apply(
                    normalized_text, context, principal_id=principal.id,
                    language=result_language or rule_language,
                )
            timings = {
                "receive_duration_ms": request.state.receive_duration_ms,
                "upload_spool_duration_ms": spool_duration,
                "routing_duration_ms": routed.routing_duration_ms,
                "server_duration_ms": milliseconds(request.state.request_started_at),
            }
            worker_timings = safe_timings(result.get("timings"))
            if worker_timings:
                timings["worker"] = worker_timings
            transcription_id = store.create_transcription(
                raw_text=raw_text,
                corrected_text=corrected_text,
                s1_cleaned_text=s1_cleaned_text,
                normalized_text=normalized_text,
                language=result_language,
                segments=list(result.get("segments") or []),
                applied_rule_ids=applied_ids,
                context=context,
                principal_id=principal.id,
                metadata={
                    "word_timestamps": word_timestamps,
                    "initial_prompt_supplied": bool(initial_prompt),
                    "keyword_count": len(keywords),
                    "original_filename": audio.filename,
                    "content_type": audio.content_type,
                    "allowed_languages": list(language_policy),
                    "detected_language": result.get("detected_language"),
                    "detected_languages": detected_languages,
                    "provider_used": routed.provider_id,
                    "requested_strategy": strategy or router.default_strategy,
                    "attempts": list(routed.attempts),
                    "degraded": routed.degraded,
                    "routing_duration_ms": routed.routing_duration_ms,
                    "usage": result.get("usage"),
                    "normalization": normalization,
                    "normalized_applied_correction_rule_ids": normalized_ids,
                    "request_id": request.state.request_id,
                    "timings": timings,
                },
            )
            return {
                "id": transcription_id,
                "request_id": request.state.request_id,
                "timings": timings,
                "raw_text": raw_text,
                "text": corrected_text,
                "normalized_text": normalized_text,
                "s1_cleaned_text": s1_cleaned_text,
                "normalization": normalization,
                "normalized_applied_correction_rule_ids": normalized_ids,
                "language": result_language,
                "detected_language": result.get("detected_language"),
                "detected_languages": detected_languages,
                "allowed_languages": list(language_policy),
                "language_policy": "restricted" if language_policy else "automatic",
                "segments": list(result.get("segments") or []),
                "applied_correction_rule_ids": applied_ids,
                "exact_override_id": exact_override_id,
                "provider_used": routed.provider_id,
                "requested_strategy": strategy or router.default_strategy,
                "attempts": list(routed.attempts),
                "degraded": routed.degraded,
                "routing_duration_ms": routed.routing_duration_ms,
                "usage": result.get("usage"),
            }

        @app.post("/v1/transcriptions/{transcription_id}/feedback", tags=["transcription"])
        async def feedback(
            request: Request,
            transcription_id: str,
            payload: FeedbackRequest,
        ) -> dict[str, Any]:
            principal = _principal(request)
            if not payload.rule_ids and payload.confirmed_text is None:
                raise HTTPException(422, "provide rule_ids and/or confirmed_text")
            attached: list[str] = []
            if payload.rule_ids:
                try:
                    result = store.attach_feedback(
                        transcription_id,
                        payload.rule_ids,
                        payload.note,
                        principal_id=principal.id,
                        include_all=principal.admin,
                    )
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
                        principal_id=principal.id,
                        include_all=principal.admin,
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
