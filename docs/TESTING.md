# CtrlSpeak Testing Playbook

CtrlSpeak has a headless suite for configuration and orchestration logic, an opt-in model/GUI integration suite, and a workstation automation flow. Run commands from the repository root in an activated environment created from `requirements.txt`.

## 1. Required development checks

Run these after every code change:

```bash
python -m compileall .
python -m pytest -m core_headless
```

The core marker covers configuration paths, CLI parsing, discovery, backend selection, remote request/feedback transport, local exact-correction persistence, non-suppressing active-field feedback capture, and v0.3 packaging metadata. It is GUI-free and does not download or load a Whisper model.

To run all headless tests regardless of marker:

```bash
python -m pytest -q tests/core
```

Useful focused commands for the v0.3 behavior are:

```bash
python -m pytest -q tests/core/test_transcription_backend.py
python -m pytest -q tests/core/test_local_corrections.py
python -m pytest -q tests/core/test_feedback_capture.py
python -m pytest -q tests/core/test_packaging_v03.py
```

## 2. Full GUI/model integration

The full test downloads/loads Whisper assets, exercises the local server, and requires the runtime GUI/audio/model dependencies:

```bash
CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest -m full_gui
```

Run the combined suite with:

```bash
CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest
```

Without `CTRLSPEAK_RUN_FULL_TESTS=1`, the integration module skips before importing the heavy model stack.

## 3. Workstation automation flow

The automation flow is an end-to-end regression harness for a provisioned Windows workstation:

```powershell
.\.venv\Scripts\python.exe main.py --automation-flow
```

It stages the selected model, validates CPU and eligible CUDA transcription, simulates text-injection strategies, and writes reports under `%APPDATA%\CtrlSpeak\automation\artifacts`. CUDA/model/network failures are recorded in `%APPDATA%\CtrlSpeak\logs\ctrlspeak.log`; a failed run leaves resumable automation state.

Run this when validating a release workstation or after changes to model/CUDA staging, real transcription, or injection behavior. It is not a substitute for the headless suite.

## 4. Whisper API service tests

The companion service has its own environment and test suite:

```bash
cd /home/trueai/services/whisper-transcription
.venv/bin/pytest
```

Those tests use a fake transcription backend. They do not download a model or require a GPU, change the bind address, or start/restart the installed service.

## 5. Optional static analysis

`pylint` and `pyright` are listed in `requirements.txt`. Examples:

```bash
python -m pylint utils/transcription_backend.py utils/local_corrections.py utils/feedback_capture.py
python -m pyright utils/transcription_backend.py utils/local_corrections.py utils/feedback_capture.py
```

Treat warnings as actionable, while recognizing that the older GUI/model modules may carry pre-existing findings outside a focused change.
