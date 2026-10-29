"""Bounded native VCDIFF encoding/application; never patches an installed file."""
from pathlib import Path
import os
import signal
import subprocess
import time

from utils.delta_tools import verified_tool

ALGORITHM = "xdelta3-vcdiff-1"
MAX_PATCH_BYTES = 128 * 1024 * 1024
MAX_PATCH_RATIO = 0.80
SOURCE_WINDOW = 64 * 1024 * 1024
TARGET_WINDOW = 8 * 1024 * 1024


class PatchError(RuntimeError):
    pass


class PatchCancelled(PatchError):
    pass


class PatchTooLarge(PatchError):
    pass


def run_tool(arguments, *, output, maximum_output, timeout=120, cancel=None):
    output = Path(output)
    if cancel is not None and cancel.is_set():
        raise PatchCancelled("Patch operation cancelled")
    environment = dict(os.environ)
    environment.pop("XDELTA", None)
    environment.pop("XDELTA3", None)
    options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == "nt" else {"start_new_session": True})
    process = None
    succeeded = False
    try:
        process = subprocess.Popen([str(verified_tool()), *map(str, arguments)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment, **options)
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if cancel is not None and cancel.is_set():
                raise PatchCancelled("Patch operation cancelled")
            if time.monotonic() >= deadline:
                raise PatchError("Patch operation exceeded its deadline")
            if output.exists() and output.stat().st_size > maximum_output:
                raise PatchTooLarge("Patch output exceeds the signed size limit")
            time.sleep(0.05)
        if cancel is not None and cancel.is_set():
            raise PatchCancelled("Patch operation cancelled")
        if output.is_file() and output.stat().st_size > maximum_output:
            raise PatchTooLarge("Patch output exceeds the signed size limit")
        if process.returncode or not output.is_file():
            raise PatchError("Patch operation failed verification")
        succeeded = True
    finally:
        if process is not None and process.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, subprocess.SubprocessError, OSError):
                pass
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
        if not succeeded:
            output.unlink(missing_ok=True)


def create_patch(base, target, output, *, timeout=180):
    base, target, output = map(Path, (base, target, output))
    if output.resolve() in {base.resolve(), target.resolve()}:
        raise PatchError("Patch output must not replace its input")
    run_tool(["-e", "-f", "-D", "-R", "-A", "-B", str(SOURCE_WINDOW),
              "-W", str(TARGET_WINDOW), "-s", base, target, output],
             output=output, maximum_output=MAX_PATCH_BYTES, timeout=timeout)


def apply_patch_file(base, patch, output, target_size, *, cancel=None, timeout=120):
    base, patch, output = map(Path, (base, patch, output))
    if output.resolve() in {base.resolve(), patch.resolve()}:
        raise PatchError("Reconstruction must not replace its input")
    # Explicit paths and -D/-R disable external compression helpers and header
    # filename selection. The caller verifies signed patch/base/target hashes.
    run_tool(["-d", "-f", "-D", "-R", "-B", str(SOURCE_WINDOW), "-s", base, patch, output],
             output=output, maximum_output=target_size, cancel=cancel, timeout=timeout)


def self_test():
    """Small real native roundtrip, no installation, network or UI input."""
    import hashlib
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ctrlspeak-delta-health-") as directory:
        root = Path(directory)
        base, target, patch, output = (root/name for name in ("base", "target", "patch", "output"))
        data = b"CtrlSpeak binary patch health\0" * 16384
        base.write_bytes(data)
        target.write_bytes(data[:80000] + b"0.8 differential update" + data[80000:])
        create_patch(base, target, patch, timeout=10)
        apply_patch_file(base, patch, output, target.stat().st_size, timeout=10)
        if hashlib.sha256(output.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
            raise PatchError("Patch runtime self-test failed")
    return {"algorithm": ALGORITHM, "ok": True, "byte_identical": True}
