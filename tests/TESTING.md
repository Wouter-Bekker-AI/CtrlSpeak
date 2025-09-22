# CtrlSpeak Test Suites

This directory hosts two pytest suites that split fast headless coverage from
GUI- and model-heavy integration checks.

## Core headless suite (`core_headless`)

* **What it covers:** configuration helpers, CLI parsing, discovery utilities,
  and other logic that can run without a GUI or large model downloads.
* **How to run:**
  ```bash
  python -m pytest -m core_headless
  ```
  Pytest is configured (via `pytest.ini`) to only discover tests inside this
  repository's `tests/` folder so the command works even inside a virtual
  environment without tripping over third-party packages.
* **Expected runtime:** a few seconds.
* **Intended environments:** any developer workstation or CI runner (Linux,
  macOS, or Windows) without GPU/audio dependencies.

## Full integration suite (`full_gui`)

* **What it covers:** downloads and activates the Whisper "small" model,
  transcribes `assets/test.wav`, starts the local server, and exercises
  discovery endpoints.
* **Prerequisites:**
  - Install the runtime dependencies in `requirements.txt`, including
    `ctranslate2`, `faster-whisper`, and audio/GUI libraries.
  - Provide network access so the Whisper checkpoint can be downloaded if not
    already cached.
  - Because the test runs the real transcription pipeline, ensure the machine
    has sufficient CPU/RAM and allow several minutes on the first run while the
    460+ MB model downloads.
* **How to run:**
  ```bash
  CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest -m full_gui
  ```
  The environment variable opt-in keeps the slow suite from running by mistake.
* **Expected runtime:** depends on download/cache state. Subsequent runs reuse
  the cached model inside the temporary config directory created by the test
  fixture.
* **Intended environments:** Windows is recommended so GUI libraries match
  production, but the suite can run on any platform that satisfies the
  dependencies.

## Running both suites together

Invoke pytest without a marker to execute everything:
```bash
CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest
```
This runs both the fast headless checks and the full integration test. Omit the
environment variable to only run the headless suite.

## Automating the suites for bots/CI

1. **Headless checks on every change.** Configure your automation (e.g., GitHub
   Actions, Azure Pipelines, Jenkins) to install dependencies and run
   `python -m pytest -m core_headless` on each push or pull request. The suite is
   fast and does not require special hardware.
2. **Full integration as an opt-in gate.** Create a separate workflow or job
   that sets `CTRLSPEAK_RUN_FULL_TESTS=1` and runs
   `python -m pytest -m full_gui`. Trigger it on a schedule (nightly) or via a
   manual dispatch to avoid blocking day-to-day development while still giving a
   faithful integration signal.
3. **Running everything after local edits.** Developers can run
   `python -m pytest -m core_headless` after making code changes and optionally
   opt into the heavier suite when testing features that touch the GUI, server,
   or transcription pipeline.

When scripting bots, prefer explicit markers so the bot can choose the correct
coverage level:

```bash
# Fast verification
python -m pytest -m core_headless

# Full system validation (requires opt-in env var)
CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest -m full_gui
```
