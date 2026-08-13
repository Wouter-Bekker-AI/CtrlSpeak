# -*- coding: utf-8 -*-
"""External executable replacement, health confirmation, and rollback."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Mapping, Optional

from utils.config_paths import get_logger
from utils.update_manager import (
    PRODUCT_ID,
    UPDATE_CHANNEL,
    UPDATE_VARIANT,
    UpdateAsset,
    UpdateError,
    UpdateTransaction,
    _atomic_write_bytes,
    _atomic_write_json,
    classify_runtime,
    get_updates_dir,
    read_transaction_journal,
    resolve_transaction,
    sha256_file,
    update_transaction_journal,
    utc_now_iso,
    verify_file,
)


logger = get_logger(__name__)

ORIGINAL_EXIT_TIMEOUT_SECONDS = 90.0
FILE_UNLOCK_TIMEOUT_SECONDS = 30.0
HEALTH_TIMEOUT_SECONDS = 90.0
HEALTH_GRACE_SECONDS = 3.0
POLL_INTERVAL_SECONDS = 0.2


def _append_update_log(transaction: UpdateTransaction, message: str) -> None:
    safe = "".join(character for character in message if character in "\t" or ord(character) >= 32)
    line = f"{utc_now_iso()} {safe[:2000]}\n"
    try:
        transaction.updater_log_path.parent.mkdir(parents=True, exist_ok=True)
        with transaction.updater_log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        logger.exception("Unable to append updater transaction log")


def _asset_from_journal(journal: Mapping[str, object]) -> UpdateAsset:
    try:
        asset = UpdateAsset(
            platform=str(journal["platform"]),
            architecture=str(journal["architecture"]),
            variant=str(journal["variant"]),
            name=str(journal["asset_name"]),
            url=str(journal["asset_url"]),
            size=int(str(journal["asset_size"])),
            sha256=str(journal["asset_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise UpdateError("invalid_transaction", "The update journal has invalid artifact metadata.") from exc
    if asset.variant != UPDATE_VARIANT:
        raise UpdateError("wrong_product", "The update transaction belongs to a different variant.")
    return asset


def _validate_transaction_identity(journal: Mapping[str, object]) -> None:
    if journal.get("product") != PRODUCT_ID:
        raise UpdateError("wrong_product", "The update transaction belongs to a different product.")
    if journal.get("channel") != UPDATE_CHANNEL:
        raise UpdateError("wrong_channel", "The update transaction belongs to a different channel.")


def resolve_transaction_argument(value: str) -> UpdateTransaction:
    candidate = Path(value)
    if candidate.name == value and len(value) == 32:
        return resolve_transaction(value)
    try:
        journal_path = candidate.resolve(strict=True)
        updates_root = get_updates_dir().resolve()
    except OSError as exc:
        raise UpdateError("missing_transaction", "The update transaction cannot be found.") from exc
    if journal_path.name != "transaction.json" or journal_path.parent.parent != updates_root:
        raise UpdateError("unsafe_path", "The update transaction path is outside CtrlSpeak update storage.")
    transaction = resolve_transaction(journal_path.parent.name)
    if transaction.journal_path.resolve() != journal_path:
        raise UpdateError("unsafe_path", "The update transaction path is invalid.")
    return transaction


def _stable_executable_name() -> str:
    return "CtrlSpeak.exe" if sys.platform.startswith("win") else "CtrlSpeak"


def _copy_verified(source: Path, destination: Path, expected_hash: str, expected_size: int) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.copy")
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if temporary.stat().st_size != expected_size or sha256_file(temporary) != expected_hash:
            raise UpdateError("copy_verification_failed", "A copied update file failed verification.")
        if not sys.platform.startswith("win"):
            temporary.chmod(0o755)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _detached_process_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True
    return kwargs


def prepare_update_handoff(
    transaction: UpdateTransaction,
    *,
    original_pid: Optional[int] = None,
    executable: Optional[Path] = None,
    launch_process: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> int:
    """Copy verified files into place and start the detached old-version helper."""

    installed = Path(executable or sys.executable).resolve()
    if classify_runtime(executable=installed) != "packaged_user_writable":
        raise UpdateError(
            "manual_install_required",
            "This CtrlSpeak copy cannot safely replace itself. Install the release manually.",
        )
    if installed.name != _stable_executable_name():
        raise UpdateError(
            "unstable_install_name",
            f"Self-update requires the stable installed filename {_stable_executable_name()}.",
        )

    journal = read_transaction_journal(transaction)
    _validate_transaction_identity(journal)
    asset = _asset_from_journal(journal)
    if journal.get("state") != "ready_to_install":
        raise UpdateError("invalid_transaction_state", "The update has not reached verified install state.")
    recorded_installation = Path(str(journal.get("installation_path", ""))).resolve()
    if recorded_installation != installed:
        raise UpdateError("installation_moved", "CtrlSpeak moved after the update download was prepared.")
    verify_file(transaction.candidate_path, asset)

    sibling_candidate = installed.with_name(installed.name + ".new")
    _copy_verified(transaction.candidate_path, sibling_candidate, asset.sha256, asset.size)

    current_size = installed.stat().st_size
    current_hash = sha256_file(installed)
    helper_name = "ctrlspeak-updater-helper.exe" if sys.platform.startswith("win") else "ctrlspeak-updater-helper"
    helper_path = transaction.directory / helper_name
    _copy_verified(installed, helper_path, current_hash, current_size)

    backup_path = transaction.directory / "previous-executable"
    health_request = {
        "schema_version": 1,
        "transaction_id": transaction.transaction_id,
        "expected_version": str(journal.get("version", "")),
        "expected_executable": str(installed),
        "requested_at": utc_now_iso(),
    }
    _atomic_write_json(transaction.health_request_path, health_request)
    updated = update_transaction_journal(
        transaction,
        state="launching_updater",
        original_pid=int(original_pid or os.getpid()),
        original_size=current_size,
        original_sha256=current_hash,
        helper_path=str(helper_path),
        sibling_candidate_path=str(sibling_candidate),
        backup_path=str(backup_path),
    )
    _append_update_log(
        transaction,
        f"Launching detached helper for {updated.get('version')} from transaction {transaction.transaction_id}",
    )
    try:
        process = launch_process(
            [str(helper_path), "--apply-update", str(transaction.journal_path)],
            cwd=str(transaction.directory),
            **_detached_process_kwargs(),
        )
    except OSError as exc:
        update_transaction_journal(transaction, state="failed", error_code="helper_launch_failed")
        raise UpdateError("helper_launch_failed", "CtrlSpeak could not start the external update helper.") from exc
    update_transaction_journal(
        transaction,
        state="awaiting_original_exit",
        helper_pid=int(process.pid),
    )
    return int(process.pid)


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _wait_for_process_exit(
    pid: int,
    timeout: float,
    *,
    process_exists: Callable[[int], bool] = _process_exists,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_exists(pid):
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    return not process_exists(pid)


def _wait_for_file_access(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with path.open("rb+"):
                return True
        except OSError:
            time.sleep(POLL_INTERVAL_SECONDS)
    return False


def _terminate_process(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            logger.exception("Unable to terminate failed post-update process")


def _valid_health_response(
    transaction: UpdateTransaction,
    journal: Mapping[str, object],
    installed: Path,
) -> bool:
    try:
        payload = json.loads(transaction.health_response_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("transaction_id") == transaction.transaction_id
        and payload.get("version") == journal.get("version")
        and Path(str(payload.get("executable", ""))).resolve() == installed
        and isinstance(payload.get("pid"), int)
    )


def _rollback(
    transaction: UpdateTransaction,
    journal: Mapping[str, object],
    installed: Path,
    new_process: Optional[subprocess.Popen],
    *,
    reason_code: str,
    launch_process: Callable[..., subprocess.Popen],
) -> None:
    _terminate_process(new_process)
    backup_path = Path(str(journal.get("backup_path", ""))).resolve()
    original_hash = str(journal.get("original_sha256", ""))
    original_size = int(str(journal.get("original_size", 0)))
    if not backup_path.is_file() or backup_path.parent != transaction.directory.resolve():
        update_transaction_journal(transaction, state="rollback_failed", error_code="missing_backup")
        raise UpdateError("rollback_failed", "The previous CtrlSpeak executable backup is missing.")
    if backup_path.stat().st_size != original_size or sha256_file(backup_path) != original_hash:
        update_transaction_journal(transaction, state="rollback_failed", error_code="invalid_backup")
        raise UpdateError("rollback_failed", "The previous CtrlSpeak executable backup is invalid.")
    rollback_sibling = installed.with_name(installed.name + ".rollback")
    _copy_verified(backup_path, rollback_sibling, original_hash, original_size)
    os.replace(rollback_sibling, installed)
    if not sys.platform.startswith("win"):
        installed.chmod(0o755)
    update_transaction_journal(
        transaction,
        state="rolled_back",
        error_code=reason_code,
        rolled_back_at=utc_now_iso(),
    )
    _append_update_log(transaction, f"Rollback completed after {reason_code}")
    try:
        launch_process(
            [str(installed), "--rollback-notice", transaction.transaction_id],
            cwd=str(installed.parent),
            **_detached_process_kwargs(),
        )
    except OSError as exc:
        raise UpdateError(
            "rollback_relaunch_failed",
            "CtrlSpeak restored the previous executable but could not relaunch it.",
        ) from exc


def apply_update_transaction(
    argument: str,
    *,
    original_exit_timeout: float = ORIGINAL_EXIT_TIMEOUT_SECONDS,
    file_unlock_timeout: float = FILE_UNLOCK_TIMEOUT_SECONDS,
    health_timeout: float = HEALTH_TIMEOUT_SECONDS,
    health_grace: float = HEALTH_GRACE_SECONDS,
    process_exists: Callable[[int], bool] = _process_exists,
    launch_process: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> int:
    """Entry point run by the copied old-version executable."""

    transaction = resolve_transaction_argument(argument)
    journal = read_transaction_journal(transaction)
    _validate_transaction_identity(journal)
    if journal.get("state") not in {"launching_updater", "awaiting_original_exit"}:
        raise UpdateError("invalid_transaction_state", "The update helper was started in an invalid state.")
    asset = _asset_from_journal(journal)
    installed = Path(str(journal.get("installation_path", ""))).resolve()
    sibling_candidate = Path(str(journal.get("sibling_candidate_path", ""))).resolve()
    backup_path = Path(str(journal.get("backup_path", ""))).resolve()
    if sibling_candidate != installed.with_name(installed.name + ".new"):
        raise UpdateError("unsafe_path", "The staged replacement path is invalid.")
    if backup_path.parent != transaction.directory.resolve():
        raise UpdateError("unsafe_path", "The rollback backup path is invalid.")
    original_pid = int(str(journal.get("original_pid", 0)))
    _append_update_log(transaction, f"Waiting for original PID {original_pid} to exit")
    if not _wait_for_process_exit(
        original_pid,
        original_exit_timeout,
        process_exists=process_exists,
    ):
        update_transaction_journal(transaction, state="failed", error_code="original_process_timeout")
        raise UpdateError("original_process_timeout", "The previous CtrlSpeak process did not exit in time.")
    if not _wait_for_file_access(installed, file_unlock_timeout):
        update_transaction_journal(transaction, state="failed", error_code="executable_locked")
        raise UpdateError("executable_locked", "The CtrlSpeak executable remained locked.")

    original_size = int(str(journal.get("original_size", 0)))
    original_hash = str(journal.get("original_sha256", ""))
    if installed.stat().st_size != original_size or sha256_file(installed) != original_hash:
        raise UpdateError("original_changed", "The installed CtrlSpeak executable changed during the update.")
    _copy_verified(installed, backup_path, original_hash, original_size)
    verify_file(sibling_candidate, asset)

    new_process: Optional[subprocess.Popen] = None
    try:
        os.replace(sibling_candidate, installed)
        if not sys.platform.startswith("win"):
            installed.chmod(0o755)
        verify_file(installed, asset)
        update_transaction_journal(
            transaction,
            state="launching_new_version",
            replaced_at=utc_now_iso(),
        )
        new_process = launch_process(
            [str(installed), "--post-update", transaction.transaction_id],
            cwd=str(installed.parent),
            **_detached_process_kwargs(),
        )
        update_transaction_journal(
            transaction,
            state="awaiting_health_confirmation",
            new_pid=int(new_process.pid),
        )
        deadline = time.monotonic() + health_timeout
        health_ok = False
        while time.monotonic() < deadline:
            if new_process.poll() is not None:
                break
            if transaction.health_response_path.is_file() and _valid_health_response(
                transaction,
                journal,
                installed,
            ):
                health_ok = True
                break
            time.sleep(POLL_INTERVAL_SECONDS)
        if health_ok:
            grace_deadline = time.monotonic() + health_grace
            while time.monotonic() < grace_deadline:
                if new_process.poll() is not None:
                    health_ok = False
                    break
                time.sleep(POLL_INTERVAL_SECONDS)
        if not health_ok:
            _rollback(
                transaction,
                journal,
                installed,
                new_process,
                reason_code="health_confirmation_failed",
                launch_process=launch_process,
            )
            return 1

        update_transaction_journal(
            transaction,
            state="installed",
            installed_at=utc_now_iso(),
            health_confirmed=True,
        )
        _append_update_log(transaction, "New version health confirmation succeeded")
        return 0
    except UpdateError:
        raise
    except OSError as exc:
        if backup_path.is_file():
            _rollback(
                transaction,
                journal,
                installed,
                new_process,
                reason_code="replacement_failed",
                launch_process=launch_process,
            )
            return 1
        raise UpdateError("replacement_failed", "CtrlSpeak could not replace its executable.") from exc


def write_post_update_health(transaction_id: str, current_version: str) -> UpdateTransaction:
    transaction = resolve_transaction(transaction_id)
    journal = read_transaction_journal(transaction)
    _validate_transaction_identity(journal)
    executable = Path(sys.executable).resolve()
    if journal.get("version") != current_version:
        raise UpdateError("health_version_mismatch", "The updated process version does not match the transaction.")
    if Path(str(journal.get("installation_path", ""))).resolve() != executable:
        raise UpdateError("health_path_mismatch", "The updated process path does not match the transaction.")
    response = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "version": current_version,
        "executable": str(executable),
        "pid": os.getpid(),
        "healthy_at": utc_now_iso(),
    }
    _atomic_write_json(transaction.health_response_path, response)
    logger.info("Wrote post-update health response for transaction %s", transaction_id)
    return transaction


def load_release_metadata(transaction_id: str) -> dict[str, object]:
    transaction = resolve_transaction(transaction_id)
    try:
        payload = json.loads(transaction.release_metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise UpdateError("missing_release_notes", "The update release information is unavailable.") from exc
    if not isinstance(payload, dict):
        raise UpdateError("missing_release_notes", "The update release information is invalid.")
    return payload


def rollback_notice(transaction_id: str) -> str:
    transaction = resolve_transaction(transaction_id)
    journal = read_transaction_journal(transaction)
    if journal.get("state") != "rolled_back":
        return "CtrlSpeak started from a previous version after an incomplete update."
    return (
        "CtrlSpeak could not confirm that the new version started safely, so the previous "
        "executable was restored automatically. Your settings and models were retained."
    )
