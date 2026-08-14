# CtrlSpeak v0.5 testing playbook

Run commands from the v0.5 repository root in a project-compatible Python
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
- ordered output-language validation, settings migration, API propagation, and
  refusal of out-of-policy API responses.
- exact-only local correction persistence.
- observer-only bare Enter handling with no suppression, replay, or duplicate.
- lazy Linux/Windows input routing and X11/Wayland capability reporting.
- preferred-xclip and no-xclip tkinter clipboard restoration/injection behavior,
  including Unicode/multiline selections and unavailable-display errors.
- Linux CUDA driver routing without Windows loader calls.
- v0.5 PyInstaller, native icons, desktop launcher, and AppStream metadata.
- preservation of historical specs while the standard helper selects v0.5.
- strict semantic versions and exact product/platform/architecture selection.
- Ed25519 manifest verification, immutable release URLs, size and SHA-256 checks.
- resumable bounded downloads, Range validation, cancellation, and oversize refusal.
- update-operation generation IDs and stale worker-event rejection.
- external replacement health confirmation and automatic rollback with dummy files.
- settings schema migration, atomic writes, per-field salvage, and backup.
- stable v0.5 Windows/Linux packaging and release-manifest tooling.

Focused commands:

```bash
python -m pytest -q tests/core/test_linux_platform.py
python -m pytest -q tests/core/test_linux_models.py
python -m pytest -q tests/core/test_packaging_v04_linux.py
python -m pytest -q tests/core/test_config_paths.py
python -m pytest -q tests/core/test_transcription_backend.py
python -m pytest -q tests/core/test_languages.py
python -m pytest -q tests/core/test_feedback_capture.py
python -m pytest -q tests/core/test_local_corrections.py
python -m pytest -q tests/core/test_update_manager.py
python -m pytest -q tests/core/test_update_helper.py
python -m pytest -q tests/core/test_release_tools.py
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
source environment and again against `dist/CtrlSpeak`:

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

## Packaged updater acceptance

The updater cannot be proven end to end by source-mode unit tests. Against the
exact signed artifacts on clean Windows and Ubuntu/X11 hosts:

1. Install v0.5.0 as `CtrlSpeak.exe` or `CtrlSpeak` and confirm the tray/control
   center show 0.5.0.
2. Publish a controlled signed v0.5.1 release with both required platform
   artifacts and the three metadata assets.
3. Check for the update from the GUI, inspect version/size, download, and confirm
   the UI remains responsive.
4. Restart and verify the same stable path now reports 0.5.1 while API URL/token,
   mode, input device, models, CUDA files, and corrections remain intact.
5. Interrupt and resume a download; confirm the final artifact hash matches the
   signed manifest.
6. Test offline, GitHub rate-limit, corrupt signature, wrong hash, and unwritable
   install-location messages without closing the current app.
7. Use a deliberately non-healthy candidate in a controlled test release and
   confirm the helper restores/relaunches the previous executable.
8. Confirm a source checkout can report the release but never enables binary
   installation.

## Packaging validation

The actual build is intentionally an operator step:

```bash
python -m utils.build_exe
test -x dist/CtrlSpeak
file dist/CtrlSpeak
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

The maintained v0.5.2 remote service is in `server/whisper_transcription`. Its
headless suite uses a fake model and does not download CUDA/model assets, bind a
network port, restart systemd, or change a firewall:

```bash
cd server/whisper_transcription
python -m pytest -q
python -m compileall app tests
```

The server tests cover the OpenAPI version/field, authentication boundary,
legacy single-language compatibility, ordered allowlist propagation, invalid
policy rejection, hard refusal of out-of-policy responses, corrections, and
confirmed-text feedback. A deployment acceptance test must additionally use
real audio against the CUDA service and assert the response language occurs in
the requested allowlist.
