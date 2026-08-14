from __future__ import annotations

import json
import logging
import mimetypes
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx


LOGGER = logging.getLogger("ctrlspeak_whisper_transcription.providers")


class InferenceBackend(Protocol):
    name: str
    device: str
    compute_type: str

    def load(self) -> None: ...

    def transcribe(
        self,
        audio_path: Path,
        allowed_languages: tuple[str, ...],
        initial_prompt: str | None,
        word_timestamps: bool,
    ) -> dict[str, Any]: ...


class ProviderFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool,
        status_code: int = 503,
        action: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code
        self.action = action

    def public_dict(self, provider_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "provider": provider_id,
            "status": "failed",
            "category": self.category,
            "retryable": self.retryable,
        }
        if self.action:
            result["action"] = self.action
        return result


@dataclass(frozen=True)
class ProviderContext:
    allowed_languages: tuple[str, ...]
    initial_prompt: str | None
    keywords: tuple[str, ...]
    word_timestamps: bool
    openai_api_key: str | None


class Provider(Protocol):
    id: str

    def describe(self) -> dict[str, Any]: ...

    def transcribe(self, audio_path: Path, context: ProviderContext) -> dict[str, Any]: ...


class LocalWhisperProvider:
    def __init__(self, provider_id: str, backend: InferenceBackend, *, lazy: bool = False) -> None:
        self.id = provider_id
        self.backend = backend
        self.lazy = lazy
        self._loaded = False
        self._load_lock = threading.Lock()

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if not self._loaded:
                self.backend.load()
                self._loaded = True

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "local_whisper",
            "model": self.backend.name,
            "device": self.backend.device,
            "compute_type": self.backend.compute_type,
            "status": "ready" if self._loaded else ("available" if self.lazy else "starting"),
            "quality": "fallback" if self.lazy else "high",
            "requires_credential": False,
            "paid": False,
        }

    def transcribe(self, audio_path: Path, context: ProviderContext) -> dict[str, Any]:
        try:
            self.load()
            return self.backend.transcribe(
                audio_path,
                context.allowed_languages,
                context.initial_prompt,
                context.word_timestamps,
            )
        except Exception as exc:
            raise ProviderFailure(
                "local transcription provider failed",
                category="local_provider_failed",
                retryable=True,
            ) from exc


class RemoteWorkerProvider:
    def __init__(
        self,
        provider_id: str,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 90.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.id = provider_id
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout_seconds = timeout_seconds
        self._client = client

    def _client_context(self):
        if self._client is not None:
            return nullcontext(self._client)
        return httpx.Client(timeout=self.timeout_seconds)

    def describe(self) -> dict[str, Any]:
        status = "configured"
        model, device, compute_type = "large-v3-turbo", "cuda", "float16"
        try:
            with self._client_context() as client:
                response = client.get(
                    f"{self.base_url}/health",
                    headers={"Authorization": f"Bearer {self._token}"},
                    timeout=min(self.timeout_seconds, 3.0),
                )
            if response.status_code == 200:
                payload = response.json()
                status = str(payload.get("status") or "ready")
                model = str(payload.get("model") or model)
                device = str(payload.get("device") or device)
                compute_type = str(payload.get("compute_type") or compute_type)
            else:
                status = "unavailable"
        except Exception:
            status = "unavailable"
        return {
            "id": self.id,
            "kind": "ctrlspeak_worker",
            "model": model,
            "device": device,
            "compute_type": compute_type,
            "status": status,
            "quality": "high",
            "requires_credential": False,
            "paid": False,
        }

    def transcribe(self, audio_path: Path, context: ProviderContext) -> dict[str, Any]:
        data: dict[str, str] = {
            "allowed_languages": ",".join(context.allowed_languages),
            "word_timestamps": str(context.word_timestamps).lower(),
            "keywords": json.dumps(context.keywords),
        }
        if context.initial_prompt:
            data["initial_prompt"] = context.initial_prompt
        content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        try:
            with audio_path.open("rb") as handle, self._client_context() as client:
                response = client.post(
                    f"{self.base_url}/v1/worker/transcribe",
                    headers={"Authorization": f"Bearer {self._token}"},
                    data=data,
                    files={"audio": (audio_path.name, handle, content_type)},
                    timeout=self.timeout_seconds,
                )
        except httpx.RequestError as exc:
            raise ProviderFailure(
                "GPU worker is unreachable",
                category="worker_unavailable",
                retryable=True,
            ) from exc
        if response.status_code != 200:
            raise ProviderFailure(
                "GPU worker rejected the transcription request",
                category="worker_error",
                retryable=response.status_code >= 500,
                status_code=502,
            )
        try:
            return dict(response.json())
        except (ValueError, TypeError) as exc:
            raise ProviderFailure(
                "GPU worker returned an invalid response",
                category="worker_invalid_response",
                retryable=True,
                status_code=502,
            ) from exc


class OpenAITranscriptionProvider:
    def __init__(
        self,
        provider_id: str = "openai-gpt-transcribe",
        *,
        model: str = "gpt-transcribe",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 90.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.id = provider_id
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._client = client

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "openai",
            "model": self.model,
            "device": "cloud",
            "compute_type": "managed",
            "status": "available_with_key",
            "quality": "high",
            "requires_credential": True,
            "credential_header": "X-CtrlSpeak-OpenAI-Key",
            "paid": True,
        }

    @staticmethod
    def _classify_error(status_code: int, payload: dict[str, Any]) -> ProviderFailure:
        error = payload.get("error") if isinstance(payload, dict) else None
        code = str(error.get("code") or "") if isinstance(error, dict) else ""
        error_type = str(error.get("type") or "") if isinstance(error, dict) else ""
        if status_code == 401:
            return ProviderFailure(
                "OpenAI rejected the supplied API key",
                category="openai_invalid_key",
                retryable=False,
                status_code=401,
                action="Check the OpenAI API key in CtrlSpeak and try again.",
            )
        if status_code in {402, 403} or code in {"insufficient_quota", "billing_hard_limit_reached"}:
            return ProviderFailure(
                "OpenAI API credit or quota is unavailable",
                category="openai_quota_exhausted",
                retryable=False,
                status_code=402,
                action="Top up or increase the limit for the supplied OpenAI account.",
            )
        if status_code == 429 and ("quota" in code or "quota" in error_type):
            return ProviderFailure(
                "OpenAI API quota is exhausted",
                category="openai_quota_exhausted",
                retryable=False,
                status_code=429,
                action="Top up or increase the limit for the supplied OpenAI account.",
            )
        return ProviderFailure(
            "OpenAI transcription is temporarily unavailable",
            category="openai_unavailable",
            retryable=status_code == 429 or status_code >= 500,
            status_code=502 if status_code >= 500 else status_code,
        )

    def transcribe(self, audio_path: Path, context: ProviderContext) -> dict[str, Any]:
        if not context.openai_api_key:
            raise ProviderFailure(
                "This route needs the caller's OpenAI API key",
                category="openai_key_required",
                retryable=True,
                status_code=422,
                action="Enter an OpenAI API key in CtrlSpeak for this session.",
            )
        form: dict[str, str | list[str]] = {
            "model": self.model,
            "response_format": "json",
        }
        if context.allowed_languages:
            form["languages[]"] = list(context.allowed_languages)
        if context.keywords:
            form["keywords[]"] = list(context.keywords)
        if context.initial_prompt:
            form["prompt"] = context.initial_prompt
        content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        try:
            client_context = (
                nullcontext(self._client)
                if self._client is not None
                else httpx.Client(timeout=self.timeout_seconds)
            )
            with audio_path.open("rb") as handle, client_context as client:
                response = client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {context.openai_api_key}"},
                    data=form,
                    files={"file": (audio_path.name, handle, content_type)},
                    timeout=self.timeout_seconds,
                )
        except httpx.RequestError as exc:
            raise ProviderFailure(
                "OpenAI transcription is unreachable",
                category="openai_unavailable",
                retryable=True,
            ) from exc
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code != 200:
            raise self._classify_error(response.status_code, payload)
        detected_languages = [
            str(item.get("code") or "").casefold()
            for item in payload.get("languages", [])
            if isinstance(item, dict) and item.get("code")
        ]
        if context.allowed_languages and any(
            code not in context.allowed_languages for code in detected_languages
        ):
            raise ProviderFailure(
                "OpenAI detected a language outside the requested allowlist",
                category="language_policy_violation",
                retryable=False,
                status_code=502,
            )
        language = detected_languages[0] if detected_languages else (
            context.allowed_languages[0] if len(context.allowed_languages) == 1 else ""
        )
        text = str(payload.get("text") or "").strip()
        return {
            "raw_text": text,
            "language": language,
            "detected_language": language or None,
            "detected_languages": detected_languages,
            "allowed_languages": list(context.allowed_languages),
            "segments": [],
            "usage": payload.get("usage"),
        }


@dataclass(frozen=True)
class RoutedResult:
    result: dict[str, Any]
    provider_id: str
    attempts: tuple[dict[str, Any], ...]
    degraded: bool


class ProviderRouter:
    def __init__(
        self,
        providers: list[Provider],
        strategies: dict[str, tuple[str, ...]],
        *,
        default_strategy: str,
    ) -> None:
        self.providers = {provider.id: provider for provider in providers}
        self.strategies = strategies
        self.default_strategy = default_strategy
        if default_strategy not in strategies:
            raise ValueError("default provider strategy is not configured")
        for strategy, chain in strategies.items():
            unknown = set(chain) - self.providers.keys()
            if unknown:
                raise ValueError(f"strategy {strategy!r} references unknown providers: {sorted(unknown)}")

    def describe(self, allowed_provider_ids: set[str] | None = None) -> dict[str, Any]:
        allowed = allowed_provider_ids or set(self.providers)
        providers = [
            provider.describe()
            for provider in self.providers.values()
            if provider.id in allowed
        ]
        strategies = [
            {"id": strategy, "providers": [item for item in chain if item in allowed]}
            for strategy, chain in self.strategies.items()
            if any(item in allowed for item in chain)
        ]
        return {
            "default_strategy": self.default_strategy,
            "providers": providers,
            "strategies": strategies,
        }

    def transcribe(
        self,
        audio_path: Path,
        context: ProviderContext,
        *,
        strategy: str | None,
        provider_id: str | None,
        allowed_provider_ids: set[str],
    ) -> RoutedResult:
        if provider_id and strategy:
            raise ProviderFailure(
                "choose either provider or strategy, not both",
                category="invalid_route",
                retryable=False,
                status_code=422,
            )
        requested = strategy or self.default_strategy
        if provider_id:
            chain = (provider_id,)
        else:
            chain = self.strategies.get(requested)
            if chain is None:
                raise ProviderFailure(
                    "unknown provider strategy",
                    category="invalid_route",
                    retryable=False,
                    status_code=422,
                )
        attempts: list[dict[str, Any]] = []
        for current_id in chain:
            if current_id not in allowed_provider_ids:
                attempts.append({"provider": current_id, "status": "not_authorized"})
                continue
            provider = self.providers.get(current_id)
            if provider is None:
                continue
            try:
                result = provider.transcribe(audio_path, context)
                attempts.append({"provider": current_id, "status": "succeeded"})
                return RoutedResult(result, current_id, tuple(attempts), len(attempts) > 1)
            except ProviderFailure as exc:
                attempts.append(exc.public_dict(current_id))
                if not exc.retryable:
                    exc.attempts = attempts  # type: ignore[attr-defined]
                    raise
        failure = ProviderFailure(
            "No configured transcription provider completed the request",
            category="providers_exhausted",
            retryable=True,
            status_code=503,
            action="Check the GPU worker or provide an OpenAI API key.",
        )
        failure.attempts = attempts  # type: ignore[attr-defined]
        raise failure
