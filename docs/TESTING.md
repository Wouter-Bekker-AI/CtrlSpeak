# CtrlSpeak Testing Playbook

CtrlSpeak ships with several test suites that cover different parts of the product. The sections below explain what each suite exercises, when to run it, and how to invoke it from a developer workstation or CI job.

## 1. Automation Flow (`python main.py --automation-flow`)

The automation flow is an end-to-end regression harness implemented in `utils/automation.py`. Running it performs the same staged checks that a fresh install would execute when provisioning a GPU-ready workstation. It is ideal for verifying that a machine can host the full CtrlSpeak stack without touching the GUI.

### What it does
1. **Model staging** - Ensures the default Whisper `small` model is present under `%APPDATA%\CtrlSpeak\models`, downloading it when missing.
2. **CUDA reuse/installation** - Copies any existing CUDA wheel assets from the active Python environment into `%APPDATA%\CtrlSpeak\cuda\12.3`. If the runtime is still not ready it installs the NVIDIA wheels (`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) and re-validates them.
3. **CPU transcription check** - Forces the device preference to CPU and transcribes the bundled `assets/test.wav` clip, logging the recognized text.
4. **GPU transcription check** - Switches to CUDA (if available) and repeats the transcription to confirm GPU inference works end-to-end.
5. **Artifact export** - Writes a consolidated transcript report to `%APPDATA%\CtrlSpeak\automation\artifacts\automation_run_YYYYMMDD-HHMMSS.txt` including the canonical transcript and simulated text-injection outputs.

Progress and failure diagnostics are recorded in `%APPDATA%\CtrlSpeak\logs\ctrlspeak.log`. If a stage fails the flow halts immediately, leaving `automation_state.json` in `%APPDATA%\CtrlSpeak\automation` so a rerun can resume from the failed step after the underlying issue is resolved.

### How to run
```powershell
# From the project root
.\.venv\Scripts\python.exe main.py --automation-flow
```

### When to run
- Provisioning or validating a new GPU workstation.
- After touching CUDA or model-staging logic (e.g., edits in `utils/automation.py`, `utils/models.py`, or management UI device actions).
- As part of a nightly regression job that exercises the full stack without manual interaction.

## 2. Core Headless Pytest Suite (`-m core_headless`)

**Coverage:** Configuration helpers, CLI parsing, discovery utilities, and other logic that requires no GUI, audio, or large downloads.

**Invocation:**
```bash
python -m pytest -m core_headless
```

**Recommended cadence:** Run after code changes during development and on every CI pull request. The suite is fast and hardware-agnostic (Windows, macOS, or Linux).

## 3. Full GUI / Integration Pytest Suite (`-m full_gui`)

**Coverage:** Downloads and activates the Whisper `small` model, exercises the hotkey transcription flow, spins up the local HTTP server, and verifies discovery endpoints.

**Prerequisites:**
- Dependencies from `requirements.txt` (including GUI/audio libraries, ctranslate2, faster-whisper).
- Network access the first time the model is downloaded.
- Sufficient CPU/RAM for the transcription pipeline.

**Invocation:**
```bash
CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest -m full_gui
```

**Recommended cadence:** Run on demand (e.g., nightly CI or before shipping features that touch the GUI, server, or transcription subsystems). Subsequent runs are faster because the model remains cached under the temporary config folder used by the tests.

## 4. Running Everything

Invoke pytest without a marker to run both suites:
```bash
CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest
```
Omit the environment variable to run only the headless suite.

## 5. Suggested Workflow for Contributors and AI Agents
- **During development:** Always run `python -m compileall .` and `python -m pytest -m core_headless`. The core headless suite must pass; if it fails, stop and document why before continuing.
- **Before major merges or releases:** Add the automation flow run and, when feasible, the full GUI suite.
- **When validating a deployment target:** Prefer the automation flow; it produces detailed artifacts and logs while avoiding manual GUI interaction.

Keeping documentation and behavior in sync is mandatory. If you change what these suites cover or how they are invoked, update this playbook alongside your code changes.

## 6. Chat With Bot Smoke Test

**Coverage:** End-to-end verification that the Chat with Bot stack initializes, transcribes audio through the CtrlSpeak server, produces an LLM reply, and synthesizes TTS audio.

**Invocation:**
```powershell
py -3 run_test_script.py
```

**What it does:**
- Launches `main.py --start-server-only` using the project virtual environment.
- Waits for `/ping` to confirm the local transcription server is ready.
- Streams `assets/test_16k_mono.wav` through SocialRobot via `utils.bot_integration.run_bot_test` and prints the bot's reply.
- Calls `/kill` to tear down the server and shows captured stdout/stderr when available.

**When to run:** Execute after touching `utils/bot_integration.py`, assets under `third_party/social_robot`, or any change that impacts server startup/transcriber initialization for the bot workflow.

## 7. Optional Static Analysis

The repository includes [pylint](https://pylint.pycqa.org/) for static code analysis. Use it when you want to perform an additional code quality pass, either on a specific file or the entire codebase. Example commands:

```powershell
.\.venv\Scripts\pylint utils/system.py
.\.venv\Scripts\pylint main.py
```

You can also lint broader sets of files:

```powershell
.\.venv\Scripts\pylint utils/*.py third_party/social_robot/**/*.py
```

Failures from pylint are not mandatory to resolve for every change, but treat warnings as actionable feedback and consider addressing them before merging significant work.
