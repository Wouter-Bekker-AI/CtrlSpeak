from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Principal:
    id: str
    admin: bool = False
    allowed_providers: tuple[str, ...] = ("*",)

    def may_use(self, provider_id: str) -> bool:
        return "*" in self.allowed_providers or provider_id in self.allowed_providers


class AuthConfigError(ValueError):
    pass


class Authenticator:
    """Resolve bearer tokens without ever retaining them in request metadata or logs."""

    def __init__(
        self,
        *,
        clients_json: str | None = None,
        legacy_token: str | None = None,
        worker_token: str | None = None,
    ) -> None:
        self._clients: list[tuple[str, Principal]] = []
        self._worker_token = worker_token or None
        if clients_json:
            self._load_clients(clients_json)
        if legacy_token:
            self._clients.append(
                (legacy_token, Principal("legacy", admin=True, allowed_providers=("*",)))
            )

    def _load_clients(self, source: str) -> None:
        try:
            decoded = json.loads(source)
        except json.JSONDecodeError as exc:
            raise AuthConfigError("CTRLSPEAK_CLIENTS_JSON must be valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise AuthConfigError("CTRLSPEAK_CLIENTS_JSON must be an object keyed by client id")
        for client_id, raw_config in decoded.items():
            if not isinstance(client_id, str) or not client_id.strip():
                raise AuthConfigError("every CtrlSpeak client id must be a non-empty string")
            if isinstance(raw_config, str):
                token = raw_config
                config: Mapping[str, Any] = {}
            elif isinstance(raw_config, Mapping):
                token = raw_config.get("token")
                config = raw_config
            else:
                raise AuthConfigError(f"client {client_id!r} must be a token or object")
            if not isinstance(token, str) or not token:
                raise AuthConfigError(f"client {client_id!r} has no non-empty token")
            providers = config.get("providers", ["*"])
            if not isinstance(providers, list) or not all(
                isinstance(item, str) and item for item in providers
            ):
                raise AuthConfigError(f"client {client_id!r} providers must be a string array")
            self._clients.append(
                (
                    token,
                    Principal(
                        client_id.strip(),
                        admin=bool(config.get("admin", False)),
                        allowed_providers=tuple(providers),
                    ),
                )
            )

    @staticmethod
    def bearer_value(authorization: str | None) -> str | None:
        scheme, separator, supplied = (authorization or "").partition(" ")
        if not separator or scheme.casefold() != "bearer" or not supplied:
            return None
        return supplied

    def authenticate_client(self, authorization: str | None) -> Principal | None:
        supplied = self.bearer_value(authorization)
        if supplied is None:
            return None
        matched: Principal | None = None
        for expected, principal in self._clients:
            if secrets.compare_digest(supplied, expected):
                matched = principal
        return matched

    def authenticate_worker(self, authorization: str | None) -> bool:
        supplied = self.bearer_value(authorization)
        return bool(
            supplied
            and self._worker_token
            and secrets.compare_digest(supplied, self._worker_token)
        )

    @property
    def has_client_credentials(self) -> bool:
        return bool(self._clients)

    @property
    def has_worker_credential(self) -> bool:
        return bool(self._worker_token)
