# CtrlSpeak v0.4 testing playbook

Run commands from the v0.4 repository root in a project-compatible Python
environment. The required fast suite is GUI-free and performs no model
download, desktop installation, service operation, or firewall change.

## Required headless checks

```bash
python -m pytest -m core_headless
python -m pytest -q tests/core
python -m compileall .
git diff --check
```

The core marker covers:

- XDG configuration paths and user-only persistence permissions.
- explicit embedded/API selection and pinned runtime configuration.
- configurable HTTP(S) URLs, optional bearer auth, API transcription IDs, and
  final-text feedback routing.
- exact-only local correction persistence.
- observer-only bare Enter handling with no suppression, replay, or duplicate.
- lazy Linux/Windows input routing and X11/Wayland capability reporting.
- preferred-xclip and no-xclip tkinter clipboard restoration/injection behavior,
  including Unicode/multiline selections and unavailable-display errors.
- Linux CUDA driver routing without Windows loader calls.
- v0.4 PyInstaller, PNG icon, desktop launcher, and AppStream metadata.
- preservation of the legacy v0.3 Windows packaging selection.

Focused commands:

```bash
python -m pytest -q tests/core/test_linux_platform.py
python -m pytest -q tests/core/test_linux_models.py
python -m pytest -q tests/core/test_packaging_v04_linux.py
python -m pytest -q tests/core/test_config_paths.py
python -m pytest -q tests/core/test_transcription_backend.py
python -m pytest -q tests/core/test_feedback_capture.py
python -m pytest -q tests/core/test_local_corrections.py
```

The tests stub optional GUI/audio packages during headless collection. Passing
them does not prove that the host has X11, PortAudio, a tray backend, or working
Whisper native libraries.

## Opt-in embedded integration

This suite downloads/loads model assets and starts the legacy local server:

```bash
CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest -m full_gui
```

Combined run:

```bash
CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest
```

Run it only in an environment intentionally provisioned for the model and
network test. It does not exercise a physical global hotkey or active third-party
application.

## Physical Ubuntu/X11 acceptance checklist

These checks must be performed by the release parent/operator against the exact
source environment and again against `dist/CtrlSpeak_v0.4`:

1. Sign into **Ubuntu on Xorg** and confirm `echo "$XDG_SESSION_TYPE"` reports
   `x11` and `DISPLAY` is set.
2. Start CtrlSpeak and verify the management window and tray menu. Confirm
   **Manage CtrlSpeak** reopens/raises the single management window and **Quit**
   stops the listener/tray cleanly.
3. Select a real microphone, hold right Ctrl, speak, release, and confirm the
   WAV is removed from the XDG temp directory after processing.
4. In embedded CPU mode, verify the existing `small` model downloads/loads from
   the XDG model directory and inserts text in at least a GTK text field, a web
   browser field, and a terminal. Record unsupported controls honestly.
5. Configure the intended LAN API URL (do not hardcode it), with and without its
   optional token as appropriate. Verify the returned ID is retained and a
   changed final field is sent to `/v1/transcriptions/{id}/feedback`.
6. With `xclip` available, put plain text on the clipboard, inject/capture once,
   and confirm the prior text returns. Repeat without `xclip` using
   Unicode/multiline clipboard and transcript text; confirm the Tk fallback
   remains invisible and restores the prior text. Put image/non-text data on
   the clipboard and confirm CtrlSpeak avoids overwriting it.
7. Edit an injected field. Verify Shift+Enter does not consume feedback, then
   bare Enter reaches the target exactly once while changed text is submitted.
8. Log into a native Wayland session and verify CtrlSpeak reports the Xorg
   requirement without crashing; do not record the global workflow as working.
9. If Linux GPU support is in release scope, provision compatible NVIDIA,
   CUDA/cuDNN, and CTranslate2 dependencies externally, use **Recheck system
   CUDA**, and compare a real CPU/GPU transcript. CtrlSpeak must not install or
   alter those system components.
10. Validate the manual `.desktop` result and icon only after placeholder
    replacement. Confirm no launcher was installed by the build itself.

## Packaging validation

The actual build is intentionally an operator step:

```bash
python -m utils.build_exe
test -x dist/CtrlSpeak_v0.4
file dist/CtrlSpeak_v0.4
```

When available:

```bash
desktop-file-validate /tmp/ctrlspeak.desktop
appstreamcli validate --no-net packaging/linux/io.trueai.CtrlSpeak.metainfo.xml
```

Also inspect the bundle's startup log at
`$XDG_CONFIG_HOME/CtrlSpeak/logs/ctrlspeak.log` (or
`~/.config/CtrlSpeak/logs/ctrlspeak.log`) for missing shared libraries/backends.

## Legacy workstation automation

`python main.py --automation-flow` remains the v0.3 provisioned Windows
workstation harness. Its SendInput and bundled CUDA-wheel stages are not a Linux
acceptance test. Do not use its host-level guidance on an Ubuntu system.

## Companion API tests

The remote service owns its own tests and deployment. Running this client suite
does not bind, restart, or reconfigure it. If an operator explicitly validates a
companion checkout, use that service's own documented environment and test
command without changing its bind address or system service.
