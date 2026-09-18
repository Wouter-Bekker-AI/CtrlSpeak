"""Run one isolated pytest process with a hard deadline and tree cleanup."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid


def main():
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--cwd", default=".")
    args, pytest_args = parser.parse_known_args()
    root = Path(args.cwd).resolve()
    base = root / ".pytest-isolated" / uuid.uuid4().hex
    base.parent.mkdir(parents=True, exist_ok=True)
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    process = subprocess.Popen([sys.executable, "-m", "pytest", "--basetemp", str(base),
                                *pytest_args], cwd=root, **options)
    try:
        return process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print("Test deadline reached; terminating and reaping this test process tree.", flush=True)
        return 124
    except KeyboardInterrupt:
        return 130
    finally:
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=10, check=False)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
