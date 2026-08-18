from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
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
from app.corrections import CorrectionStore
from app.languages import choose_allowed_language, normalize_language_policy
from app.providers import (
    LocalWhisperProvider,
    OpenAITranscriptionProvider,
    ProviderContext,
    ProviderFailure,
    ProviderRouter,
    RemoteWorkerProvider,
)


SERVICE_VERSION = "0.7.2"
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
) -> FastAPI:
    environ = os.environ if environ is None else environ
    role = _service_role(service_role or environ.get("CTRLSPEAK_SERVICE_ROLE"))
    data_dir = Path(environ.get("WHISPER_DATA_DIR", str(DEFAULT_DATA_DIR)))
    store = None if role == "worker" else (store or CorrectionStore(data_dir / "corrections.sqlite3"))
    backend = backend or _build_default_backend(role, data_dir)
    temp_dir = temp_dir or data_dir / "uploads"
    clients_json = clients_json if clients_json is not None else environ.get("CTRLSPEAK_CLIENTS_JSON")
    worker_token = worker_token if worker_token is not None else environ.get("CTRLSPEAK_WORKER_TOKEN")
    authenticator = Authenticator(
        clients_json=clients_json,
        legacy_token=bearer_token,
        worker_token=worker_token,
    )
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

    app = FastAPI(
        title="CtrlSpeak Transcription API",
        version=SERVICE_VERSION,
        description=(
            "Role-aware CtrlSpeak transcription service. Gateways route requests and apply "
            "identity-scoped corrections; workers return raw model output only."
        ),
        lifespan=lifespan,
    )
    app.state.role = role
    app.state.router = router

    @app.middleware("http")
    async def require_bearer_auth(request: Request, call_next):
        client_host = request.client.host if request.client else None
        if _is_loopback_client(client_host):
            request.state.principal = Principal("legacy", admin=True)
            request.state.worker_authenticated = True
            return await call_next(request)

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
            return await call_next(request)

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
        return await call_next(request)

    @app.get("/health", tags=["service"])
    async def health(request: Request) -> dict[str, Any]:
        if not request.app.state.ready:
            raise HTTPException(503, "service is not ready")
        payload: dict[str, Any] = {
            "status": "ready",
            "version": SERVICE_VERSION,
            "role": role,
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
                "accepts_client_transcriptions": False,
                "applies_corrections": False,
                "telemetry": {
                    "inference_duration_ms": True,
                },
                "provider": {
                    "id": "ubuntu-gpu-large-v3-turbo",
                    "kind": "local_whisper",
                    "model": getattr(backend, "name", "unknown"),
                    "device": getattr(backend, "device", "unknown"),
                    "compute_type": getattr(backend, "compute_type", "unknown"),
                    "status": "ready",
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
            "accepts_client_transcriptions": True,
            "applies_corrections": True,
            "openai_key_storage": "request_only",
            "telemetry": {
                "attempt_duration_ms": True,
                "routing_duration_ms": True,
                "worker_inference_duration_ms": True,
                "worker_health": True,
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
        except Exception:
            if temp_path:
                temp_path.unlink(missing_ok=True)
            raise

    if role == "worker":

        @app.post("/v1/worker/transcribe", tags=["worker"])
        async def worker_transcribe(
            audio: UploadFile = File(...),
            allowed_languages: str | None = Form(None),
            language: str | None = Form(None),
            initial_prompt: str | None = Form(None),
            keywords: str | None = Form(None),
            word_timestamps: bool = Form(False),
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
            temp_path = await receive_audio(audio)
            inference_started_at = time.monotonic()
            try:
                result = await run_in_threadpool(
                    backend.transcribe,
                    temp_path,
                    language_policy,
                    effective_prompt,
                    word_timestamps,
                )
                inference_duration_ms = round(
                    max(0.0, time.monotonic() - inference_started_at) * 1000.0,
                    3,
                )
            except Exception as exc:
                LOGGER.exception("worker transcription failed")
                raise HTTPException(500, "worker transcription failed") from exc
            finally:
                await audio.close()
                temp_path.unlink(missing_ok=True)
            result["provider"] = "ubuntu-gpu-large-v3-turbo"
            result["inference_duration_ms"] = inference_duration_ms
            result["corrected"] = False
            return result

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
            )
            temp_path = await receive_audio(audio)
            try:
                routed = await run_in_threadpool(
                    router.transcribe,
                    temp_path,
                    provider_context,
                    strategy=strategy,
                    provider_id=provider,
                    allowed_provider_ids=_allowed_provider_ids(principal, router),
                )
            except ProviderFailure as exc:
                detail: dict[str, Any] = {
                    "message": str(exc),
                    "category": exc.category,
                    "retryable": exc.retryable,
                    "attempts": getattr(exc, "attempts", []),
                }
                routing_duration_ms = getattr(exc, "routing_duration_ms", None)
                if routing_duration_ms is not None:
                    detail["routing_duration_ms"] = routing_duration_ms
                if exc.action:
                    detail["action"] = exc.action
                raise HTTPException(exc.status_code, detail) from exc
            finally:
                await audio.close()
                temp_path.unlink(missing_ok=True)

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
            transcription_id = store.create_transcription(
                raw_text=raw_text,
                corrected_text=corrected_text,
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
                },
            )
            return {
                "id": transcription_id,
                "raw_text": raw_text,
                "text": corrected_text,
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
