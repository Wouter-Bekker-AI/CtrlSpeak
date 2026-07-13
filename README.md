# CtrlSpeak v0.4 for Ubuntu/Linux

CtrlSpeak is a native desktop speech-to-text client. Hold the **right Ctrl**
key to record, release it to transcribe, and CtrlSpeak inserts the result into
the active field. v0.4 adds a maintained Ubuntu/Linux path while preserving the
v0.3 backend design:

- **Embedded / local** runs the bundled `faster-whisper` workflow and retains
  the existing model (`small` or `large-v3`) and device (`cpu` or `cuda`)
  settings.
- **Remote API** sends the recording to a configurable HTTP(S) base URL with an
  optional bearer token. It retains the API transcription ID and submits an
  edited final result to that same API endpoint.

The legacy **Client + Server** and **Client Only** roles remain inside the
embedded/local backend. Remote API mode is independent of those roles and does
not start discovery, a local server, or a model download.

## Linux support boundary

The desktop hotkey, active-field capture, and text-injection workflow is
supported on an **X11/Xorg session**. On Ubuntu's sign-in screen, choose the
gear icon and **Ubuntu on Xorg** before signing in.

Native Wayland global input is not implemented. CtrlSpeak detects Wayland,
leaves the listener stopped, and presents an actionable message instead of
crashing or pretending that an XWayland window makes system-wide input work.
Headless shells and services are also unsupported because there is no active
field or desktop keyboard session.

On X11:

- `pynput` observes the global right-Ctrl hotkey and bare Enter. The listener is
  always configured with `suppress=False`.
- `pyautogui` performs X11 keystrokes.
- `xclip` is the preferred clipboard provider for paste and feedback capture.
  CtrlSpeak restores the prior text clipboard after the operation.
- If `xclip` is absent, the included tkinter fallback uses a withdrawn Tk root,
  services X11 selection events while it owns staged text, and restores the
  prior text before cleaning up. It supports Unicode and multiline text.
- If neither `xclip` nor the Tk/X11 fallback is usable, CtrlSpeak reports the
  display/runtime problem; plain single-line ASCII retains its direct-typing
  last resort.
- If the clipboard advertises non-text formats, CtrlSpeak does not overwrite
  them. It uses the same ASCII fallback when possible; otherwise insertion or
  feedback capture is skipped with a logged explanation.

Some protected, elevated, custom-rendered, remote, terminal, password, or
multiline controls may reject Ctrl+A/C or expose incomplete text. Correction
capture is therefore best effort, not universal accessibility-API support.

## Ubuntu prerequisites and source setup

Use a project virtual environment. Typical Ubuntu build/runtime prerequisites
are:

```bash
sudo apt install python3-venv python3-dev python3-tk portaudio19-dev \
  libportaudio2 xclip libx11-6 libxtst6 libxinerama1 libxrandr2 libxi6
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`xclip` remains preferred, but it is optional at runtime when tkinter can open
the active X11 display. `python3-tk` supplies that included fallback for source
environments; the Linux bundle must include the same tkinter runtime support.

Ubuntu GNOME may also require its AppIndicator support/extension before a
legacy tray icon is visible. CtrlSpeak keeps Tk on the main thread and runs the
Linux tray backend in its dedicated event thread. If tray startup fails, the
application reports the failure and opens the management window so settings and
Quit remain reachable. `pystray`'s Xorg fallback has limited shell integration;
GTK/AppIndicator support may require PyGObject and desktop-specific packages.

PyAudio uses the host's PortAudio input devices. Microphone permission, default
source selection, PipeWire/PulseAudio compatibility, and per-device levels are
operator/desktop settings; CtrlSpeak does not alter them.

Run from source:

```bash
. .venv/bin/activate
python main.py
```

Runtime state is never written beside the source tree or packaged executable.
It lives at:

```text
$XDG_CONFIG_HOME/CtrlSpeak/
```

If `XDG_CONFIG_HOME` is unset, the path is `~/.config/CtrlSpeak`. This directory
contains `settings.json`, `models/`, `cuda/`, `temp/`, `logs/`, the instance
lock, Hugging Face cache, and `local-corrections.sqlite3`. Settings and local
correction files are created with user-only POSIX permissions.

## Transcription backends

The defaults are:

```json
{
  "transcription_backend": "bundled",
  "api_url": "http://127.0.0.1:8765",
  "api_token": null,
  "feedback_capture_method": "active_field_on_enter",
  "device_preference": "cpu",
  "model_name": "small"
}
```

The API URL is a complete configurable `http://` or `https://` base URL. It may
point to loopback, a LAN/VPN host, or an HTTPS service. No current LAN address
is hardcoded. API mode calls:

- `POST <base-url>/v1/transcribe` with multipart WAV audio.
- `POST <base-url>/v1/transcriptions/{id}/feedback` with the confirmed final
  text, capture method, and client audit metadata.

The transcribe response must include non-empty `text`, `raw_text`, and `id`.
CtrlSpeak retains those fields plus the complete response metadata. A pending
feedback item remains bound to the original URL and in-memory token even if
saved settings are later changed. Network, HTTP, and response-schema failures
are actionable and never silently fall back to local transcription.

An optional token is sent as `Authorization: Bearer <token>`. There is no token
CLI flag because command arguments leak through process listings and shell
history. A token saved through the management window is plain text in the
user-only settings file. Runtime status displays only whether a token exists.

Environment overrides are:

```bash
export CTRLSPEAK_BACKEND=api
export CTRLSPEAK_API_URL=https://whisper.example.test/base
export CTRLSPEAK_API_TOKEN='...'
python main.py
```

Backend settings are pinned at startup. Saving a backend, URL, token, or
feedback-method change in the management window requires a restart, preventing
a partially switched runtime.

## Embedded/local Whisper

Embedded mode preserves model download, storage, selection, CPU inference, and
the legacy client/server behavior. The default `small` model is downloaded on
first embedded launch and stored under the XDG CtrlSpeak directory. Local
approved edits create exact raw-transcript-to-final-text overrides in
`local-corrections.sqlite3`; similar transcripts are not broadly rewritten.

CPU is the safe default. On Linux, CtrlSpeak can select `cuda` when
`libcuda.so.1` reports an NVIDIA device and CTranslate2 can use the installed
CUDA/cuDNN runtime. CtrlSpeak v0.4 does **not** install Linux GPU drivers or
system CUDA libraries. The management window offers **Recheck system CUDA**;
the Windows wheel-based installer remains Windows-only. `--download-cuda-only`
therefore validates Linux system readiness and reports the missing operator
prerequisites rather than modifying the host.

## Edit feedback and Enter behavior

With `active_field_on_enter` enabled:

1. CtrlSpeak injects one transcription.
2. The user edits the active field.
3. The user presses bare Enter normally.
4. Before that original non-suppressed Enter reaches the application, CtrlSpeak
   best-effort snapshots the field with Ctrl+A/C, restores the prior text
   clipboard, compares the result, and schedules feedback only if it changed.

CtrlSpeak never suppresses, replays, synthesizes, or duplicates Enter. Modified
Enter (for example Shift+Enter), key releases, and unrelated keys do not consume
the pending item. Pending state is single-use, is replaced by the next
injection, and expires after ten minutes. Set the feedback method to `disabled`
when whole-field selection is unsuitable.

## CLI

```text
--backend {bundled,api}       Persist the backend
--api-url <http(s)-url>       Persist the complete API base URL
--backend-status              Print redacted backend status and exit
--auto-setup {client,client_server}
--transcribe <wav>            Transcribe a file
--download-cuda-only          Windows install / Linux readiness check
--setup-cuda                  Alias for --download-cuda-only
--force-sendinput             Windows-only SendInput preference
--automation-flow             Legacy provisioned-workstation harness
--uninstall                   Automatic self-removal on Windows only
```

Use `python main.py --help` for parser details. On Linux, remove a manually
installed executable, icon, and desktop file manually; CtrlSpeak does not run a
package manager or delete arbitrary installation paths.

## Linux packaging and launcher template

The maintained Linux build command is:

```bash
python -m utils.build_exe
```

It selects `packaging/CtrlSpeak_v0.4.spec` and produces the clearly named
one-file executable `dist/CtrlSpeak_v0.4` on a Linux build host. The PNG icon and
runtime assets are bundled; model weights and system CUDA libraries remain
external. PyInstaller does not cross-compile this artifact from Windows.

The repository includes:

- `packaging/linux/ctrlspeak.desktop` — uninstalled launcher template.
- `packaging/linux/io.trueai.ctrlspeak.metainfo.xml` — AppStream metadata.
- `assets/icon.png` — native Linux application icon.

Detailed prerequisites, placeholder replacement, optional user-local install
commands, and post-build verification are in `packaging/BUILDING.md`. Nothing in
the build helper installs a desktop file, starts a service, changes a firewall,
or deploys an API.

## Development checks

```bash
python -m pytest -m core_headless
python -m pytest -q tests/core
python -m compileall .
git diff --check
```

The full embedded model/server integration remains opt-in because it downloads
and loads Whisper assets:

```bash
CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest -m full_gui
```

See `docs/TESTING.md` for focused commands and the required physical Ubuntu
desktop checks. A headless suite cannot prove microphone capture, GNOME tray
visibility, global X11 hooks, active-application injection, or model/GPU
performance.

## Network and security boundary

Configuring a URL does not bind, expose, restart, deploy, or reconfigure the
remote service. CtrlSpeak does not change services or firewall policy. Use HTTPS
or a trusted VPN for untrusted networks, keep tokens out of source and command
lines, and protect transcription/correction data according to its sensitivity.

The legacy embedded `/transcribe` plus UDP-discovery protocol is intended for a
controlled trusted LAN and has no bearer-auth contract. Do not publish it to the
Internet.

## License

MIT License. Copyright (c) 2025 CtrlSpeak contributors.
