"""Minimal Goose headless invocation helper."""

import os
import subprocess
import sys
import textwrap

ALLOWED_MODES = {"auto", "smart_approve", "approve", "chat"}


def goose_query(
    prompt: str,
    *,
    model: str = "qwen3:14b",
    mode: str = "auto",
    provider: str = "ollama",
    goose_exe: str = "goose",
    stream: bool = False,
) -> str:
    """Run Goose CLI once and return the raw output, optionally streaming live."""
    if prompt is None:
        raise ValueError("prompt must be provided")

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")

    normalized_mode = mode.strip().lower()
    if normalized_mode not in ALLOWED_MODES:
        raise ValueError("mode must be one of 'auto', 'smart_approve', 'approve', or 'chat'")

    instructions = textwrap.dedent(
        f"""
        You are an on-machine agent. Use tools if needed (for example, the developer builtin shell).
        When you are completely finished, respond with a single line JSON object: {{"final": "<answer>"}}.
        Do not include commentary, code fences, or analysis outside of that JSON line.

        User request:
        {prompt}
        """
    ).strip()

    command = [
        goose_exe,
        "run",
        "--text",
        instructions,
        "--no-session",
        "--with-builtin",
        "developer",
        "--provider",
        provider,
        "--model",
        model,
        "--max-turns",
        "40",
        "--max-tool-repetitions",
        "3",
    ]

    env = os.environ.copy()
    env["GOOSE_MODE"] = normalized_mode

    if stream:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            env=env,
        )
        assert proc.stdout is not None
        collected: list[str] = []
        for line in proc.stdout:
            sys.stdout.buffer.write(line.encode("utf-8", "replace"))
            sys.stdout.buffer.flush()
            collected.append(line)
        proc.wait()
        returncode = proc.returncode
        output = "".join(collected)
    else:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            env=env,
        )
        returncode = completed.returncode
        output = completed.stdout or ""
        if output:
            sys.stdout.buffer.write(output.encode("utf-8", "replace"))
            sys.stdout.buffer.flush()

    if returncode != 0:
        raise RuntimeError(f"Goose failed (exit {returncode}).\n{output.strip()}")

    if not output.strip():
        raise RuntimeError("Goose produced no output.")

    return output
