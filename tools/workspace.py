"""Workspace tooling utilities for Einstein.

This module exposes a narrow, auditable tool surface for file discovery,
inspection, editing, and validation within the CtrlSpeak repository. All
functions operate inside a sandboxed workspace root and return structured
results that callers can serialise directly.

The helpers favour safety over flexibility: paths are normalised, SHA-256
hashes gate writes, and JSON/text edits are expressed as patches. Validators
perform lightweight parsing by default and opt into stricter checks when the
relevant third-party packages are available.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import py_compile
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.io_atomic import atomic_write_text


class WorkspaceToolError(RuntimeError):
    """Raised when a workspace tool encounters an unrecoverable problem."""


class Sha256MismatchError(WorkspaceToolError):
    """Raised when a caller supplies an out-of-date SHA-256 digest."""

    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(f"SHA-256 mismatch (expected {expected}, found {actual})")
        self.expected = expected
        self.actual = actual


_DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.environ.get("CTRLSPK_WORKSPACE_ROOT", _DEFAULT_WORKSPACE_ROOT)).resolve()

_EXTRA_ALLOWED_ROOTS: set[Path] = set()
_ALLOWED_ROOTS: set[Path] = set()


def _refresh_allowed_roots() -> None:
    _ALLOWED_ROOTS.clear()
    _ALLOWED_ROOTS.add(WORKSPACE_ROOT)
    _ALLOWED_ROOTS.update(_EXTRA_ALLOWED_ROOTS)




def set_workspace_root(path: str | Path) -> Path:
    """Override the workspace root. Intended for tests."""

    global WORKSPACE_ROOT
    WORKSPACE_ROOT = Path(path).expanduser().resolve()
    _refresh_allowed_roots()
    return WORKSPACE_ROOT


def register_allowed_root(path: str | Path) -> List[str]:
    """Allow workspace tools to operate within an additional filesystem root."""

    resolved = Path(path).expanduser().resolve(strict=False)
    if resolved not in _EXTRA_ALLOWED_ROOTS:
        _EXTRA_ALLOWED_ROOTS.add(resolved)
        _refresh_allowed_roots()
    return [root.as_posix() for root in sorted(_ALLOWED_ROOTS)]


def set_additional_allowed_roots(paths: Iterable[str | Path]) -> List[str]:
    """Replace the set of non-workspace allowed roots."""

    _EXTRA_ALLOWED_ROOTS.clear()
    for entry in paths:
        resolved = Path(entry).expanduser().resolve(strict=False)
        _EXTRA_ALLOWED_ROOTS.add(resolved)
    _refresh_allowed_roots()
    return [root.as_posix() for root in sorted(_ALLOWED_ROOTS)]


def list_allowed_roots() -> List[str]:
    """Return all filesystem roots accessible to the workspace tools."""

    return [root.as_posix() for root in sorted(_ALLOWED_ROOTS)]


def list_additional_roots() -> List[str]:
    """Return the non-workspace roots that have been explicitly registered."""

    return [root.as_posix() for root in sorted(_EXTRA_ALLOWED_ROOTS)]


_refresh_allowed_roots()

try:
    _home_root = Path.home().expanduser().resolve()
except Exception:
    _home_root = None
else:
    register_allowed_root(_home_root)
    anchor = _home_root.anchor
    if anchor and os.name == "nt":
        try:
            register_allowed_root(Path(anchor).resolve(strict=False))
        except Exception:
            register_allowed_root(Path(anchor))

for env_var in ("APPDATA", "LOCALAPPDATA", "USERPROFILE", "HOMEDRIVE"):
    value = os.environ.get(env_var)
    if not value:
        continue
    try:
        register_allowed_root(Path(value).expanduser().resolve(strict=False))
    except Exception:
        try:
            register_allowed_root(Path(value).expanduser())
        except Exception:
            continue


def _relative_path(path: Path) -> str:
    candidates = [WORKSPACE_ROOT, *_ALLOWED_ROOTS]
    seen = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def _resolve_path(path: str | Path) -> Path:
    if isinstance(path, Path):
        raw = str(path)
    else:
        raw = str(path)

    if not raw:
        raise WorkspaceToolError("Path must be a non-empty string.")

    normalized = raw.replace("\\", "/")
    if normalized.startswith("~"):
        raise WorkspaceToolError("Home-relative paths are not allowed; provide a path under the workspace root.")

    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate

    resolved = candidate.resolve(strict=False)
    for root in _ALLOWED_ROOTS:
        try:
            if resolved.is_relative_to(root):
                return resolved
        except ValueError:
            continue
    allowed_list = ", ".join(str(root) for root in sorted(_ALLOWED_ROOTS)) or str(WORKSPACE_ROOT)
    raise WorkspaceToolError(
        f"Path {path} is not under an allowed root ({allowed_list})."
    )


def _sha256_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ok(**payload: Any) -> Dict[str, Any]:
    result = {"ok": True, "errors": []}
    result.update(payload)
    return result


def _error(message: str, *, code: str = "error", **extra: Any) -> Dict[str, Any]:
    payload = {"ok": False, "errors": [{"code": code, "message": message}]}
    payload.update(extra)
    return payload


def search_files(query: str, glob: str | None = None, *, limit: int = 200) -> Dict[str, Any]:
    """Return files whose relative path matches ``query`` using an optional glob."""

    if not query:
        return _error("Query must be a non-empty string.")

    def _compile_pattern(source: str) -> re.Pattern[str]:
        try:
            return re.compile(source, re.IGNORECASE)
        except re.error:
            escaped = re.escape(source)
            return re.compile(escaped, re.IGNORECASE)

    normalized_query = query.replace("\\", "/")
    patterns: List[re.Pattern[str]] = [_compile_pattern(query)]
    if normalized_query != query:
        patterns.append(_compile_pattern(normalized_query))

    matches: List[str] = []
    seen: set[str] = set()

    search_glob = (glob or "**/*").replace("\\", "/")

    for root in sorted(_ALLOWED_ROOTS):
        try:
            resolved_root = root if root.exists() else root.resolve(strict=False)
        except Exception:
            resolved_root = root

        if not resolved_root.exists():
            continue

        root_glob = search_glob
        try:
            pattern_path = Path(search_glob)
            if pattern_path.is_absolute():
                try:
                    root_glob = pattern_path.relative_to(resolved_root).as_posix() or "**/*"
                except ValueError:
                    continue
        except Exception:
            root_glob = search_glob

        try:
            iterator = resolved_root.glob(root_glob)
        except ValueError:
            return _error(f"Invalid glob pattern: {glob}")
        except Exception:
            iterator = resolved_root.glob("**/*")

        for path in iterator:
            if not path.is_file():
                continue
            rel = _relative_path(path)
            if rel in seen:
                continue
            rel_windows = rel.replace("/", "\\")
            if any(pattern.search(candidate) for candidate in (rel, rel_windows) for pattern in patterns):
                matches.append(rel)
                seen.add(rel)
                if len(matches) >= limit:
                    return _ok(paths=matches, count=len(matches))

    return _ok(paths=matches, count=len(matches))


def list_directory(
    path: str | Path,
    *,
    pattern: str | None = None,
    extensions: Iterable[str] | None = None,
    recursive: bool = False,
    limit: int | None = 200,
) -> Dict[str, Any]:
    try:
        target = _resolve_path(path)
        if not target.exists():
            raise WorkspaceToolError(f"Directory {target} does not exist.")
        if not target.is_dir():
            raise WorkspaceToolError(f"Path {target} is not a directory.")

        normalized_pattern = (pattern or "*").replace("\\", "/")
        iterator = target.rglob(normalized_pattern) if recursive else target.glob(normalized_pattern)

        extension_set: set[str] | None = None
        if extensions:
            extension_set = {
                str(ext).lower().lstrip(".")
                for ext in extensions
                if isinstance(ext, str) and ext.strip()
            }
            if not extension_set:
                extension_set = None

        entries: List[str] = []
        truncated = False
        for entry in iterator:
            try:
                resolved_entry = entry.resolve(strict=False)
            except Exception:
                resolved_entry = entry
            if resolved_entry.is_dir():
                continue
            if extension_set is not None:
                suffix = resolved_entry.suffix.lower().lstrip(".")
                if suffix not in extension_set:
                    continue
            entries.append(_relative_path(resolved_entry))
            if limit is not None and len(entries) >= max(limit, 0):
                truncated = True
                break

        message = f"Found {len(entries)} item(s) in {target.as_posix()}."
        if truncated:
            message = f"Found at least {len(entries)} item(s) in {target.as_posix()} (results truncated)."

        return _ok(
            path=_relative_path(target),
            entries=entries,
            count=len(entries),
            truncated=truncated,
            message=message,
        )
    except WorkspaceToolError as exc:
        return _error(str(exc))


def stat_file(path: str | Path) -> Dict[str, Any]:
    """Return file existence and metadata."""

    try:
        target = _resolve_path(path)
    except WorkspaceToolError as exc:
        return _error(str(exc))

    exists = target.exists()
    info: Dict[str, Any] = {"exists": exists, "is_file": target.is_file() if exists else False}
    if exists and target.is_file():
        stat_result = target.stat()
        info.update({"size": int(stat_result.st_size), "mtime": float(stat_result.st_mtime)})
    return _ok(**info)


def _load_yaml_module():
    spec = importlib.util.find_spec("yaml")
    if spec is None:
        raise WorkspaceToolError("PyYAML is required for YAML operations but is not installed.")
    return importlib.import_module("yaml")


def _ensure_jsonschema():
    spec = importlib.util.find_spec("jsonschema")
    if spec is None:
        raise WorkspaceToolError("jsonschema is required for schema validation but is not installed.")
    return importlib.import_module("jsonschema")


_TOML_MODULE = None


def _load_toml_module():
    global _TOML_MODULE
    if _TOML_MODULE is not None:
        return _TOML_MODULE

    spec = importlib.util.find_spec("tomllib")
    if spec is not None:
        _TOML_MODULE = importlib.import_module("tomllib")
        return _TOML_MODULE

    legacy_spec = importlib.util.find_spec("tomli")
    if legacy_spec is not None:
        _TOML_MODULE = importlib.import_module("tomli")
        return _TOML_MODULE

    raise WorkspaceToolError("tomllib (Python 3.11+) or tomli is required for TOML operations but is not installed.")


def _read_bytes(path: Path) -> Tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, _sha256_digest(payload)


def read_file(path: str | Path, *, mode: str = "text", encoding: str = "utf-8") -> Dict[str, Any]:
    """Read a file relative to the workspace root."""

    try:
        target = _resolve_path(path)
    except WorkspaceToolError as exc:
        return _error(str(exc))

    if not target.is_file():
        return _error(f"File {target} does not exist or is not a regular file.")

    try:
        payload, digest = _read_bytes(target)
    except Exception as exc:  # pragma: no cover - propagated I/O issues
        return _error(f"Failed to read {target}: {exc}")

    rel = _relative_path(target)
    if mode == "bytes":
        return _ok(path=rel, content=payload, sha256=digest)
    text: Optional[str] = None
    if mode in {"text", "json", "yaml", "toml"}:
        text = payload.decode(encoding)
    if mode == "text":
        return _ok(path=rel, content=text, sha256=digest)
    if mode == "json":
        try:
            data = json.loads(text)  # type: ignore[arg-type]
        except json.JSONDecodeError as exc:
            return _error(f"Failed to parse JSON: {exc}")
        return _ok(path=rel, content=data, sha256=digest)
    if mode == "yaml":
        try:
            yaml = _load_yaml_module()
            data = yaml.safe_load(text)
        except Exception as exc:
            return _error(f"Failed to parse YAML: {exc}")
        return _ok(path=rel, content=data, sha256=digest)
    if mode == "toml":
        try:
            toml_module = _load_toml_module()
            data = toml_module.loads(text or "")
        except Exception as exc:
            return _error(f"Failed to parse TOML: {exc}")
        return _ok(path=rel, content=data, sha256=digest)
    return _error(f"Unsupported mode '{mode}'.")


def _pointer_tokens(pointer: str) -> List[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise WorkspaceToolError(f"Invalid JSON Pointer '{pointer}'.")
    parts = pointer.lstrip("/").split("/")
    return [part.replace("~1", "/").replace("~0", "~") for part in parts]


def _traverse(container: Any, tokens: List[str], *, create_missing: bool = False) -> Any:
    current = container
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                if create_missing:
                    current[token] = {}
                else:
                    raise WorkspaceToolError(f"Key '{token}' not found during traversal.")
            current = current[token]
        elif isinstance(current, list):
            if token == "-":
                raise WorkspaceToolError("'-' is only valid for the final JSON Pointer token.")
            try:
                index = int(token)
            except ValueError as exc:
                raise WorkspaceToolError(f"List index '{token}' is not an integer.") from exc
            if index < 0 or index >= len(current):
                raise WorkspaceToolError(f"List index {index} out of range.")
            current = current[index]
        else:
            raise WorkspaceToolError("Cannot traverse into a non-container value.")
    return current


def _json_patch(document: Any, operations: Iterable[Dict[str, Any]]) -> Any:
    data = copy.deepcopy(document)
    for op in operations:
        if not isinstance(op, dict):
            raise WorkspaceToolError("Each patch operation must be a JSON object.")
        operation = str(op.get("op", "")).lower()
        if not operation:
            raise WorkspaceToolError("Patch operation missing 'op' field.")
        path = op.get("path")
        if not isinstance(path, str):
            raise WorkspaceToolError("Patch operation requires a string 'path'.")
        tokens = _pointer_tokens(path)

        if operation == "add":
            value = copy.deepcopy(op.get("value"))
            if not tokens:
                data = value
                continue
            parent = _traverse(data, tokens[:-1], create_missing=True)
            last = tokens[-1]
            if isinstance(parent, list):
                if last == "-":
                    parent.append(value)
                else:
                    try:
                        index = int(last)
                    except ValueError as exc:
                        raise WorkspaceToolError(f"List index '{last}' is not an integer.") from exc
                    if index < 0 or index > len(parent):
                        raise WorkspaceToolError(f"List index {index} out of range for add.")
                    parent.insert(index, value)
            elif isinstance(parent, dict):
                parent[last] = value
            else:
                raise WorkspaceToolError("Add operation requires a dict or list target.")
            continue

        if operation == "remove":
            if not tokens:
                raise WorkspaceToolError("Remove operation cannot target the document root.")
            parent = _traverse(data, tokens[:-1])
            last = tokens[-1]
            if isinstance(parent, list):
                try:
                    index = int(last)
                except ValueError as exc:
                    raise WorkspaceToolError(f"List index '{last}' is not an integer.") from exc
                if index < 0 or index >= len(parent):
                    raise WorkspaceToolError(f"List index {index} out of range for remove.")
                parent.pop(index)
            elif isinstance(parent, dict):
                if last not in parent:
                    raise WorkspaceToolError(f"Key '{last}' not present for remove.")
                parent.pop(last)
            else:
                raise WorkspaceToolError("Remove operation requires a dict or list target.")
            continue

        if operation == "replace":
            if not tokens:
                data = copy.deepcopy(op.get("value"))
                continue
            parent = _traverse(data, tokens[:-1])
            last = tokens[-1]
            value = copy.deepcopy(op.get("value"))
            if isinstance(parent, list):
                try:
                    index = int(last)
                except ValueError as exc:
                    raise WorkspaceToolError(f"List index '{last}' is not an integer.") from exc
                if index < 0 or index >= len(parent):
                    raise WorkspaceToolError(f"List index {index} out of range for replace.")
                parent[index] = value
            elif isinstance(parent, dict):
                if last not in parent:
                    raise WorkspaceToolError(f"Key '{last}' not present for replace.")
                parent[last] = value
            else:
                raise WorkspaceToolError("Replace operation requires a dict or list target.")
            continue

        if operation == "move":
            from_path = op.get("from")
            if not isinstance(from_path, str):
                raise WorkspaceToolError("Move operation requires a string 'from' field.")
            value = _json_pointer_get(data, from_path)
            data = _json_patch(data, [{"op": "remove", "path": from_path}])
            data = _json_patch(data, [{"op": "add", "path": path, "value": value}])
            continue

        if operation == "copy":
            from_path = op.get("from")
            if not isinstance(from_path, str):
                raise WorkspaceToolError("Copy operation requires a string 'from' field.")
            value = _json_pointer_get(data, from_path)
            data = _json_patch(data, [{"op": "add", "path": path, "value": value}])
            continue

        if operation == "test":
            expected = copy.deepcopy(op.get("value"))
            actual = _json_pointer_get(data, path)
            if actual != expected:
                raise WorkspaceToolError(f"Test operation failed at '{path}'.")
            continue

        raise WorkspaceToolError(f"Unsupported patch operation '{operation}'.")

    return data


def _json_pointer_get(document: Any, pointer: str) -> Any:
    tokens = _pointer_tokens(pointer)
    if not tokens:
        return copy.deepcopy(document)
    parent = _traverse(document, tokens[:-1]) if len(tokens) > 1 else document
    last = tokens[-1]
    if isinstance(parent, dict):
        if last not in parent:
            raise WorkspaceToolError(f"Key '{last}' not found for pointer '{pointer}'.")
        return copy.deepcopy(parent[last])
    if isinstance(parent, list):
        if last == "-":
            raise WorkspaceToolError("'-' is invalid in JSON Pointer dereference.")
        try:
            index = int(last)
        except ValueError as exc:
            raise WorkspaceToolError(f"List index '{last}' is not an integer.") from exc
        if index < 0 or index >= len(parent):
            raise WorkspaceToolError(f"List index {index} out of range for pointer '{pointer}'.")
        return copy.deepcopy(parent[index])
    raise WorkspaceToolError("JSON Pointer does not reference a container.")


def _render_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _prepare_json_patch(
    path: str | Path,
    patch: Iterable[Dict[str, Any]],
    *,
    expect_sha256: Optional[str] = None,
    schema_path: str | Path | None = None,
) -> Tuple[Path, Dict[str, Any]]:
    target = _resolve_path(path)
    if not target.is_file():
        raise WorkspaceToolError(f"File {target} does not exist.")
    raw_bytes, current_sha = _read_bytes(target)
    if expect_sha256 and expect_sha256 != current_sha:
        raise Sha256MismatchError(expect_sha256, current_sha)
    try:
        document = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkspaceToolError(f"Failed to parse JSON: {exc}") from exc
    new_document = _json_patch(document, patch)
    if schema_path is not None:
        schema_file = _resolve_path(schema_path)
        schema_bytes, _ = _read_bytes(schema_file)
        try:
            schema = json.loads(schema_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkspaceToolError(f"Failed to parse schema JSON: {exc}") from exc
        jsonschema = _ensure_jsonschema()
        jsonschema.validate(new_document, schema)
    return target, {"document": document, "new_document": new_document, "current_sha": current_sha}


def dry_run_json_patch(
    path: str | Path,
    patch: Iterable[Dict[str, Any]],
    *,
    expect_sha256: Optional[str] = None,
    schema_path: str | Path | None = None,
) -> Dict[str, Any]:
    try:
        target, payload = _prepare_json_patch(
            path,
            patch,
            expect_sha256=expect_sha256,
            schema_path=schema_path,
        )
        rendered = _render_json(payload["new_document"])
        new_sha = _sha256_digest(rendered.encode("utf-8"))
        return _ok(
            path=_relative_path(target),
            preview_json=payload["new_document"],
            preview_text=rendered,
            current_sha256=payload["current_sha"],
            new_sha256=new_sha,
        )
    except Sha256MismatchError as exc:
        return _error(str(exc), code="sha_mismatch", actual_sha=exc.actual)
    except WorkspaceToolError as exc:
        return _error(str(exc))


def apply_json_patch(
    path: str | Path,
    patch: Iterable[Dict[str, Any]],
    *,
    expect_sha256: Optional[str] = None,
    schema_path: str | Path | None = None,
) -> Dict[str, Any]:
    try:
        target, payload = _prepare_json_patch(
            path,
            patch,
            expect_sha256=expect_sha256,
            schema_path=schema_path,
        )
        rendered = _render_json(payload["new_document"])
        new_sha = _sha256_digest(rendered.encode("utf-8"))
        atomic_write_text(target, rendered)
        return _ok(
            path=_relative_path(target),
            new_sha256=new_sha,
            previous_sha256=payload["current_sha"],
        )
    except Sha256MismatchError as exc:
        return _error(str(exc), code="sha_mismatch", actual_sha=exc.actual)
    except WorkspaceToolError as exc:
        return _error(str(exc))


@dataclass
class _Hunk:
    start_src: int
    len_src: int
    lines: List[str]


_HUNK_HEADER = re.compile(r"@@ -(?P<src_line>\d+)(?:,(?P<src_count>\d+))? \+(?P<dst_line>\d+)(?:,(?P<dst_count>\d+))? @@")


def _parse_unified_diff(diff: str) -> List[_Hunk]:
    lines = diff.splitlines()
    hunks: List[_Hunk] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("---") or line.startswith("+++"):
            index += 1
            continue
        match = _HUNK_HEADER.match(line)
        if not match:
            index += 1
            continue
        src_line = int(match.group("src_line"))
        src_count = int(match.group("src_count") or "1")
        index += 1
        hunk_lines: List[str] = []
        while index < len(lines) and not lines[index].startswith("@@ "):
            hunk_lines.append(lines[index])
            index += 1
        hunks.append(_Hunk(start_src=src_line, len_src=src_count, lines=hunk_lines))
    return hunks


def _apply_unified_diff(original: str, diff: str) -> str:
    original_lines = original.splitlines(keepends=True)
    result: List[str] = []
    cursor = 0
    hunks = _parse_unified_diff(diff)
    for hunk in hunks:
        start_index = max(hunk.start_src - 1, 0)
        while cursor < start_index and cursor < len(original_lines):
            result.append(original_lines[cursor])
            cursor += 1
        for entry in hunk.lines:
            if entry == "\\ No newline at end of file":
                continue
            if not entry:
                continue
            prefix = entry[0]
            content = entry[1:]
            if prefix == ' ':
                if cursor >= len(original_lines):
                    raise WorkspaceToolError("Diff context exceeds original file length.")
                reference = original_lines[cursor]
                if reference.rstrip("\n\r") != content.rstrip("\n\r"):
                    raise WorkspaceToolError("Diff context does not match original content.")
                result.append(reference)
                cursor += 1
            elif prefix == '-':
                if cursor >= len(original_lines):
                    raise WorkspaceToolError("Removal exceeds original file length.")
                reference = original_lines[cursor]
                if reference.rstrip("\n\r") != content.rstrip("\n\r"):
                    raise WorkspaceToolError("Diff removal does not match original content.")
                cursor += 1
            elif prefix == '+':
                newline = "\n"
                if cursor > 0 and original_lines[cursor - 1].endswith("\r\n"):
                    newline = "\r\n"
                result.append(f"{content}{newline}")
            else:
                raise WorkspaceToolError(f"Unsupported diff prefix '{prefix}'.")
    result.extend(original_lines[cursor:])
    return "".join(result)


def dry_run_text_patch(
    path: str | Path,
    unified_diff: str,
    *,
    expect_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        target = _resolve_path(path)
        if not target.is_file():
            raise WorkspaceToolError(f"File {target} does not exist.")
        payload, current_sha = _read_bytes(target)
        if expect_sha256 and expect_sha256 != current_sha:
            raise Sha256MismatchError(expect_sha256, current_sha)
        original_text = payload.decode("utf-8")
        patched = _apply_unified_diff(original_text, unified_diff)
        new_bytes = patched.encode("utf-8")
        new_sha = _sha256_digest(new_bytes)
        return _ok(
            path=_relative_path(target),
            preview_text=patched,
            current_sha256=current_sha,
            new_sha256=new_sha,
        )
    except Sha256MismatchError as exc:
        return _error(str(exc), code="sha_mismatch", actual_sha=exc.actual)
    except WorkspaceToolError as exc:
        return _error(str(exc))
    except UnicodeDecodeError as exc:
        return _error(f"Failed to decode file as UTF-8: {exc}")


def apply_text_patch(
    path: str | Path,
    unified_diff: str,
    *,
    expect_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    dry_run = dry_run_text_patch(path, unified_diff, expect_sha256=expect_sha256)
    if not dry_run.get("ok"):
        return dry_run
    try:
        target = _resolve_path(path)
        atomic_write_text(target, dry_run["preview_text"])
        return _ok(
            path=dry_run.get("path"),
            new_sha256=dry_run.get("new_sha256"),
            previous_sha256=dry_run.get("current_sha256"),
        )
    except WorkspaceToolError as exc:
        return _error(str(exc))


def analyze_python(path: str | Path) -> Dict[str, Any]:
    try:
        target = _resolve_path(path)
        if not target.is_file():
            raise WorkspaceToolError(f"File {target} does not exist.")
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports: List[str] = []
        functions: List[str] = []
        classes: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}:{alias.name}" if module else alias.name)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        symbols = sorted(set(functions + classes))
        return _ok(
            path=_relative_path(target),
            imports=sorted(set(imports)),
            functions=functions,
            classes=classes,
            symbols=symbols,
        )
    except WorkspaceToolError as exc:
        return _error(str(exc))
    except (UnicodeDecodeError, SyntaxError) as exc:
        return _error(f"Failed to analyse Python file: {exc}")


def validate_json(path: str | Path, *, schema_path: str | Path | None = None) -> Dict[str, Any]:
    try:
        target = _resolve_path(path)
        if not target.is_file():
            raise WorkspaceToolError(f"File {target} does not exist.")
        data_bytes, _ = _read_bytes(target)
        decoded = data_bytes.decode("utf-8")
        parsed = json.loads(decoded)
        if schema_path is not None:
            schema = _resolve_path(schema_path)
            schema_data, _ = _read_bytes(schema)
            schema_obj = json.loads(schema_data.decode("utf-8"))
            jsonschema = _ensure_jsonschema()
            jsonschema.validate(parsed, schema_obj)
        return _ok()
    except WorkspaceToolError as exc:
        return _error(str(exc))
    except json.JSONDecodeError as exc:
        return _error(f"JSON parsing failed: {exc}")
    except Exception as exc:
        return _error(f"Schema validation failed: {exc}")


def validate_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        target = _resolve_path(path)
        if not target.is_file():
            raise WorkspaceToolError(f"File {target} does not exist.")
        module = _load_yaml_module()
        text = target.read_text(encoding="utf-8")
        module.safe_load(text)
        return _ok()
    except WorkspaceToolError as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"YAML parsing failed: {exc}")


def validate_python(path: str | Path, *, mode: str = "fast") -> Dict[str, Any]:
    try:
        target = _resolve_path(path)
        if not target.is_file():
            raise WorkspaceToolError(f"File {target} does not exist.")
        py_compile.compile(str(target), doraise=True)
        warnings: List[str] = []
        if mode == "strict":
            spec = importlib.util.find_spec("ruff")
            if spec is None:
                warnings.append("ruff not installed; skipped linting.")
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "ruff", "check", "--select", "E9,F63,F7,F82", str(target)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    message = result.stderr.strip() or result.stdout.strip() or "ruff check failed"
                    return _error(message, code="lint_failed")
        return _ok(warnings=warnings)
    except WorkspaceToolError as exc:
        return _error(str(exc))
    except py_compile.PyCompileError as exc:
        return _error(f"py_compile failed: {exc}")


def format_file(path: str | Path) -> Dict[str, Any]:
    try:
        target = _resolve_path(path)
        if not target.is_file():
            raise WorkspaceToolError(f"File {target} does not exist.")
        suffix = target.suffix.lower()
        if suffix != ".py":
            return _ok(changed=False, formatter=None)
        spec = importlib.util.find_spec("black")
        if spec is None:
            return _ok(changed=False, formatter=None)
        black = importlib.import_module("black")
        source = target.read_text(encoding="utf-8")
        try:
            formatted = black.format_file_contents(source, fast=False, mode=black.FileMode())
        except black.NothingChanged:  # type: ignore[attr-defined]
            return _ok(changed=False, formatter="black")
        atomic_write_text(target, formatted)
        return _ok(changed=True, formatter="black")
    except WorkspaceToolError as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Failed to format file: {exc}")


def get_system_info() -> Dict[str, Any]:
    """Return host operating-system information for platform-aware planning."""

    system = platform.system()
    return _ok(
        platform=sys.platform,
        system=system,
        release=platform.release(),
        version=platform.version(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        is_windows=system.lower() == "windows",
        is_linux=system.lower() == "linux",
        is_macos=system.lower() == "darwin",
    )


__all__ = [
    "WORKSPACE_ROOT",
    "WorkspaceToolError",
    "Sha256MismatchError",
    "search_files",
    "stat_file",
    "read_file",
    "dry_run_json_patch",
    "apply_json_patch",
    "dry_run_text_patch",
    "apply_text_patch",
    "analyze_python",
    "validate_json",
    "validate_yaml",
    "validate_python",
    "format_file",
    "get_system_info",
    "set_workspace_root",
    "register_allowed_root",
    "set_additional_allowed_roots",
    "list_allowed_roots",
    "list_additional_roots",
]
