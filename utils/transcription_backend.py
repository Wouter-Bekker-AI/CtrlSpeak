"""Selection and HTTP transport for CtrlSpeak transcription backends.

This module stays independent of the model and GUI modules so backend selection,
status text, and HTTP behavior remain straightforward to test headlessly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import quote, urlsplit, urlunsplit

from utils import config_paths
from utils.languages import language_policy_display, normalize_allowed_output_languages


DEFAULT_API_URL = "http://127.0.0.1:8765"
VALID_BACKENDS = {"bundled", "api"}
VALID_FEEDBACK_CAPTURE_METHODS = {"disabled", "active_field_on_enter"}
BACKEND_DISPLAY_NAMES = {
    "bundled": "Embedded / local",
    "api": "Remote API",
}


class ApiBackendError(RuntimeError):
    """An actionable failure returned by, or communicating with, the API backend."""


class BackendPersistenceError(RuntimeError):
    """A validated backend configuration could not be persisted."""


class HttpSession(Protocol):
    def post(self, url: str, **kwargs: Any) -> Any: ...
    def get(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class BackendConfig:
    backend: str
    api_url: str
    api_token: str | None = dataclass_field(repr=False)
    feedback_capture_method: str
    allowed_output_languages: tuple[str, ...] = ()
    provider_strategy: str = "server-default"


@dataclass(frozen=True)
class FeedbackTarget:
    """Immutable route back to the backend that produced one transcription."""

    backend: str
    api_url: str | None = None
    api_token: str | None = dataclass_field(default=None, repr=False)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    transcription_id: str | None = None
    raw_text: str | None = None
    corrected_text: str | None = None
    metadata: Mapping[str, Any] | None = None
    feedback_target: FeedbackTarget | None = None


_runtime_backend_config: BackendConfig | None = None
_session_openai_api_key: str | None = None


def set_session_openai_api_key(value: str | None) -> None:
    """Keep the caller's OpenAI key in this process only; never write it to settings."""
    global _session_openai_api_key
    _session_openai_api_key = value.strip() if value and value.strip() else None


def has_session_openai_api_key() -> bool:
    return bool(_session_openai_api_key)


def backend_display_name(backend: str) -> str:
    try:
        return BACKEND_DISPLAY_NAMES[backend]
    except KeyError as exc:
        raise ValueError(f"unknown transcription backend: {backend}") from exc


def backend_from_display_name(display_name: str) -> str:
    normalized = display_name.strip().casefold()
    for backend, label in BACKEND_DISPLAY_NAMES.items():
        if normalized == label.casefold():
            return backend
    raise ValueError(f"unknown transcription backend display name: {display_name}")


def _normalized_choice(value: object, valid: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in valid else default


def _validated_backend_choice(value: object, *, source: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in VALID_BACKENDS:
        raise ValueError(f"{source} must be 'bundled' or 'api'; received {value!r}")
    return normalized


def _normalized_capture_method(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "clipboard_on_enter":
        # Migrate the review-build setting to the automatic v0.3 workflow.
        return "active_field_on_enter"
    return normalized


def normalize_api_url(value: object) -> str:
    candidate = str(value or "").strip().rstrip("/")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API URL must be an http:// or https:// URL with a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API URL must not contain credentials, a query, or a fragment")
    try:
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ValueError(f"API URL has an invalid host or port: {exc}") from exc
    if not hostname:
        raise ValueError("API URL must include a valid host")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def get_backend_config(environ: Mapping[str, str] | None = None) -> BackendConfig:
    env = os.environ if environ is None else environ
    with config_paths.settings_lock:
        saved = dict(config_paths.settings)

    backend_from_environment = "CTRLSPEAK_BACKEND" in env
    backend = _validated_backend_choice(
        env.get("CTRLSPEAK_BACKEND", saved.get("transcription_backend")),
        source="CTRLSPEAK_BACKEND" if backend_from_environment else "transcription_backend setting",
    )
    token_value = env.get("CTRLSPEAK_API_TOKEN")
    if token_value is None:
        token_value = saved.get("api_token")
    api_token = str(token_value).strip() if token_value else None
    api_url = normalize_api_url(
        env.get("CTRLSPEAK_API_URL", saved.get("api_url", DEFAULT_API_URL))
    )
    feedback_capture_method = _normalized_choice(
        _normalized_capture_method(saved.get("feedback_capture_method")),
        VALID_FEEDBACK_CAPTURE_METHODS,
        "active_field_on_enter",
    )
    language_source = (
        env.get("CTRLSPEAK_OUTPUT_LANGUAGES")
        if "CTRLSPEAK_OUTPUT_LANGUAGES" in env
        else saved.get("allowed_output_languages", [])
    )
    allowed_output_languages = normalize_allowed_output_languages(
        language_source,
        source=(
            "CTRLSPEAK_OUTPUT_LANGUAGES"
            if "CTRLSPEAK_OUTPUT_LANGUAGES" in env
            else "allowed_output_languages setting"
        ),
    )
    provider_strategy = str(
        env.get(
            "CTRLSPEAK_PROVIDER_STRATEGY",
            saved.get("provider_strategy", "server-default"),
        )
        or "server-default"
    ).strip()
    if not provider_strategy:
        provider_strategy = "server-default"
    return BackendConfig(
        backend,
        api_url,
        api_token,
        feedback_capture_method,
        allowed_output_languages,
        provider_strategy,
    )


def activate_runtime_backend_config(config: BackendConfig | None = None) -> BackendConfig:
    """Pin backend routing for this process; later saved changes need a restart."""
    global _runtime_backend_config
    _runtime_backend_config = config or get_backend_config()
    return _runtime_backend_config


def get_runtime_backend_config() -> BackendConfig:
    """Return the startup-pinned backend, or current config before app startup."""
    return _runtime_backend_config or get_backend_config()


def save_backend_config(
    *,
    backend: str,
    api_url: str,
    api_token: str | None,
    feedback_capture_method: str,
    allowed_output_languages: object | None = None,
    provider_strategy: str = "server-default",
) -> BackendConfig:
    normalized_backend = _validated_backend_choice(backend, source="backend")
    normalized_capture = _normalized_choice(
        _normalized_capture_method(feedback_capture_method),
        VALID_FEEDBACK_CAPTURE_METHODS,
        "",
    )
    if not normalized_capture:
        raise ValueError(
            "feedback capture method must be 'disabled' or 'active_field_on_enter'"
        )
    normalized_token = api_token.strip() if api_token and api_token.strip() else None
    normalized_url = normalize_api_url(api_url)
    with config_paths.settings_lock:
        current_languages = config_paths.settings.get("allowed_output_languages", [])
    normalized_languages = normalize_allowed_output_languages(
        current_languages if allowed_output_languages is None else allowed_output_languages,
        source="allowed output languages",
    )
    normalized_strategy = str(provider_strategy or "server-default").strip()
    if not normalized_strategy:
        raise ValueError("provider strategy must be non-empty")
    with config_paths.settings_lock:
        previous = dict(config_paths.settings)
        config_paths.settings.update(
            transcription_backend=normalized_backend,
            api_url=normalized_url,
            api_token=normalized_token,
            feedback_capture_method=normalized_capture,
            allowed_output_languages=list(normalized_languages),
            provider_strategy=normalized_strategy,
        )
    if not config_paths.save_settings():
        with config_paths.settings_lock:
            config_paths.settings.clear()
            config_paths.settings.update(previous)
        raise BackendPersistenceError(
            f"Backend settings could not be saved to {config_paths.get_config_file_path()}. "
            "Check that the CtrlSpeak configuration directory is writable and try again."
        )
    return BackendConfig(
        normalized_backend,
        normalized_url,
        normalized_token,
        normalized_capture,
        normalized_languages,
        normalized_strategy,
    )


def get_backend_status(config: BackendConfig | None = None) -> str:
    current = config or get_backend_config()
    if current.feedback_capture_method == "active_field_on_enter":
        feedback = "automatic active-field capture on Enter"
    else:
        feedback = "disabled"
    languages = language_policy_display(current.allowed_output_languages)
    if current.backend == "bundled":
        return (
            f"Embedded / local · bundled CtrlSpeak model · feedback: {feedback} "
            f"· output languages: {languages}"
        )
    auth = "bearer token configured" if current.api_token else "no bearer token"
    return (
        f"API · {current.api_url} · {auth} · feedback: {feedback} "
        f"· output languages: {languages}"
    )


def uses_bundled_runtime(config: BackendConfig | None = None) -> bool:
    return (config or get_runtime_backend_config()).backend == "bundled"


class ApiTranscriptionClient:
    def __init__(
        self,
        config: BackendConfig,
        *,
        session: HttpSession | None = None,
        timeout_seconds: float = 300.0,
        openai_api_key: str | None = None,
    ) -> None:
        self.config = config
        if session is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - packaging/runtime guard
                raise RuntimeError("The requests package is required for API transcription mode") from exc
            session = requests.Session()
        self.session = session
        self.timeout_seconds = timeout_seconds
        supplied_key = openai_api_key if openai_api_key is not None else _session_openai_api_key
        self._openai_api_key = supplied_key.strip() if supplied_key else None

    def _headers(self) -> dict[str, str]:
        if not self.config.api_token:
            return {}
        return {"Authorization": f"Bearer {self.config.api_token}"}

    @staticmethod
    def _response_detail(response: Any) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("detail"):
                detail = payload["detail"]
                if isinstance(detail, dict):
                    message = str(detail.get("message") or detail.get("category") or detail)
                    action = detail.get("action")
                    return f"{message} {action}".strip() if action else message
                return str(detail)
        except Exception:
            pass
        return str(getattr(response, "text", ""))[:400].strip() or "no response detail"

    def _post(
        self,
        path: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        url = f"{self.config.api_url}{path}"
        headers = self._headers()
        headers.update(extra_headers or {})
        try:
            response = self.session.post(
                url,
                headers=headers,
                timeout=self.timeout_seconds,
                **kwargs,
            )
        except Exception as exc:
            raise ApiBackendError(
                f"Could not reach the configured Whisper API at {self.config.api_url}. "
                "Check the URL, service status, network, and bearer-token configuration. "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc
        if response.status_code < 200 or response.status_code >= 300:
            detail = self._response_detail(response)
            raise ApiBackendError(f"Whisper API returned HTTP {response.status_code}: {detail}")
        try:
            payload = response.json()
        except Exception as exc:
            raise ApiBackendError("Whisper API returned an invalid JSON response") from exc
        if not isinstance(payload, dict):
            raise ApiBackendError("Whisper API returned a JSON response with the wrong shape")
        return payload

    def get_capabilities(self) -> dict[str, Any]:
        url = f"{self.config.api_url}/v1/capabilities"
        try:
            response = self.session.get(
                url,
                headers=self._headers(),
                timeout=min(self.timeout_seconds, 10.0),
            )
        except Exception as exc:
            raise ApiBackendError(
                f"Could not read capabilities from {self.config.api_url}: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise ApiBackendError(
                f"CtrlSpeak API returned HTTP {response.status_code}: "
                f"{self._response_detail(response)}"
            )
        try:
            payload = response.json()
        except Exception as exc:
            raise ApiBackendError("CtrlSpeak API returned invalid capability JSON") from exc
        if not isinstance(payload, dict):
            raise ApiBackendError("CtrlSpeak capability response has the wrong shape")
        if payload.get("accepts_client_transcriptions") is not True:
            raise ApiBackendError(
                "The configured endpoint is a worker, not a client-facing CtrlSpeak gateway"
            )
        return payload

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        path = Path(audio_path)
        try:
            with path.open("rb") as audio_file:
                request: dict[str, Any] = {
                    "files": {"audio": (path.name, audio_file, "audio/wav")},
                }
                request_data: dict[str, str] = {}
                if self.config.allowed_output_languages:
                    request_data.update({
                        "allowed_languages": ",".join(self.config.allowed_output_languages),
                    })
                if self.config.provider_strategy != "server-default":
                    request_data["strategy"] = self.config.provider_strategy
                if request_data:
                    request["data"] = request_data
                secret_headers = (
                    {"X-CtrlSpeak-OpenAI-Key": self._openai_api_key}
                    if self._openai_api_key
                    else None
                )
                payload = self._post(
                    "/v1/transcribe",
                    extra_headers=secret_headers,
                    **request,
                )
        except ApiBackendError:
            raise
        except OSError as exc:
            raise ApiBackendError(f"Unable to read audio file {path}: {exc}") from exc

        text = payload.get("text")
        transcription_id = payload.get("id")
        raw_text = payload.get("raw_text")
        if not isinstance(text, str) or not text.strip():
            raise ApiBackendError("Whisper API response did not include non-empty corrected text")
        if not isinstance(transcription_id, str) or not transcription_id:
            raise ApiBackendError("Whisper API response did not include a transcription id")
        if not isinstance(raw_text, str):
            raise ApiBackendError("Whisper API response did not include raw transcript metadata")
        response_language = str(payload.get("language") or "").strip().casefold()
        if (
            self.config.allowed_output_languages
            and response_language not in self.config.allowed_output_languages
        ):
            raise ApiBackendError(
                "Whisper API returned a language outside the configured output-language allowlist"
            )
        return TranscriptionResult(
            text=text,
            transcription_id=transcription_id,
            raw_text=raw_text,
            corrected_text=text,
            metadata=dict(payload),
            feedback_target=FeedbackTarget(
                backend="api",
                api_url=self.config.api_url,
                api_token=self.config.api_token,
            ),
        )

    def submit_feedback(
        self,
        transcription_id: str,
        *,
        final_text: str,
        capture_method: str,
        client_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not transcription_id:
            raise ValueError("transcription_id is required")
        if not final_text.strip():
            raise ValueError("final_text must be non-empty")
        return self._post(
            f"/v1/transcriptions/{quote(transcription_id, safe='')}/feedback",
            json={
                "confirmed_text": final_text,
                "capture_method": capture_method,
                "client_metadata": dict(client_metadata or {}),
            },
        )


def transcribe_selected(
    audio_path: Path,
    *,
    config: BackendConfig,
    bundled_transcriber: Callable[[Path], str | None],
    api_client: ApiTranscriptionClient | None = None,
    local_correction_library: Any | None = None,
) -> TranscriptionResult | None:
    """Use only the explicitly selected backend; API failures never trigger fallback."""
    if config.backend == "api":
        return (api_client or ApiTranscriptionClient(config)).transcribe(audio_path)
    text = bundled_transcriber(audio_path)
    if not text:
        return None
    if local_correction_library is None:
        from utils.local_corrections import get_local_correction_library

        local_correction_library = get_local_correction_library()
    local_result = local_correction_library.record_transcription(
        text,
        metadata={"backend": "bundled"},
    )
    return TranscriptionResult(
        text=local_result.corrected_text,
        transcription_id=local_result.transcription_id,
        raw_text=local_result.raw_text,
        corrected_text=local_result.corrected_text,
        metadata={
            "backend": "bundled",
            "exact_override_id": local_result.exact_override_id,
        },
        feedback_target=FeedbackTarget(backend="bundled"),
    )
