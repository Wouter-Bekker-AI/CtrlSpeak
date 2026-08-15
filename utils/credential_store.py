"""OS-backed storage for user-owned CtrlSpeak credentials.

OpenAI keys are never written to ``settings.json``.  On Windows they are kept
in the current user's Windows Credential Manager vault.  Other platforms stay
session-only until an equivalent native secret-store backend is implemented.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Protocol


OPENAI_CREDENTIAL_TARGET = "CtrlSpeak/OpenAI/APIKey"
_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2
_ERROR_NOT_FOUND = 1168
_MAX_CREDENTIAL_BYTES = 2560


class CredentialStorageError(RuntimeError):
    """A native credential-store operation failed without exposing the secret."""


class _CredentialBackend(Protocol):
    def read(self, target: str) -> str | None: ...
    def write(self, target: str, value: str) -> None: ...
    def delete(self, target: str) -> None: ...


class _CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


class _WindowsCredentialBackend:
    def __init__(self) -> None:
        try:
            self._advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
        except (AttributeError, OSError) as exc:
            raise CredentialStorageError("Windows Credential Manager is unavailable") from exc

        self._cred_write = self._advapi32.CredWriteW
        self._cred_write.argtypes = [ctypes.POINTER(_CREDENTIALW), wintypes.DWORD]
        self._cred_write.restype = wintypes.BOOL

        self._cred_read = self._advapi32.CredReadW
        self._cred_read.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.POINTER(_CREDENTIALW)),
        ]
        self._cred_read.restype = wintypes.BOOL

        self._cred_delete = self._advapi32.CredDeleteW
        self._cred_delete.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self._cred_delete.restype = wintypes.BOOL

        self._cred_free = self._advapi32.CredFree
        self._cred_free.argtypes = [ctypes.c_void_p]
        self._cred_free.restype = None

    @staticmethod
    def _raise(operation: str) -> None:
        error_code = ctypes.get_last_error()
        raise CredentialStorageError(
            f"Windows Credential Manager could not {operation} the CtrlSpeak credential "
            f"(Windows error {error_code})"
        )

    def read(self, target: str) -> str | None:
        pointer = ctypes.POINTER(_CREDENTIALW)()
        if not self._cred_read(target, _CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
            if ctypes.get_last_error() == _ERROR_NOT_FOUND:
                return None
            self._raise("read")
        try:
            credential = pointer.contents
            raw = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
            return raw.decode("utf-8").strip() or None
        except (UnicodeDecodeError, ValueError) as exc:
            raise CredentialStorageError(
                "The stored CtrlSpeak credential is unreadable; forget and replace it"
            ) from exc
        finally:
            self._cred_free(pointer)

    def write(self, target: str, value: str) -> None:
        encoded = value.encode("utf-8")
        if not encoded or len(encoded) > _MAX_CREDENTIAL_BYTES:
            raise CredentialStorageError("The OpenAI API key has an invalid size")
        blob = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)
        credential = _CREDENTIALW()
        credential.Type = _CRED_TYPE_GENERIC
        credential.TargetName = target
        credential.CredentialBlobSize = len(encoded)
        credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
        credential.Persist = _CRED_PERSIST_LOCAL_MACHINE
        credential.UserName = "CtrlSpeak"
        if not self._cred_write(ctypes.byref(credential), 0):
            self._raise("save")

    def delete(self, target: str) -> None:
        if self._cred_delete(target, _CRED_TYPE_GENERIC, 0):
            return
        if ctypes.get_last_error() != _ERROR_NOT_FOUND:
            self._raise("delete")


_backend: _CredentialBackend | None = None
_backend_initialized = False


def _get_backend() -> _CredentialBackend | None:
    global _backend, _backend_initialized
    if not _backend_initialized:
        _backend_initialized = True
        if sys.platform.startswith("win"):
            _backend = _WindowsCredentialBackend()
    return _backend


def secure_storage_available() -> bool:
    try:
        return _get_backend() is not None
    except CredentialStorageError:
        return False


def load_openai_api_key() -> str | None:
    backend = _get_backend()
    return backend.read(OPENAI_CREDENTIAL_TARGET) if backend is not None else None


def save_openai_api_key(value: str) -> None:
    normalized = value.strip()
    if not normalized:
        raise CredentialStorageError("The OpenAI API key cannot be empty")
    backend = _get_backend()
    if backend is None:
        raise CredentialStorageError("Secure credential storage is unavailable on this platform")
    backend.write(OPENAI_CREDENTIAL_TARGET, normalized)


def delete_openai_api_key() -> None:
    backend = _get_backend()
    if backend is not None:
        backend.delete(OPENAI_CREDENTIAL_TARGET)
