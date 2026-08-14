# CtrlSpeak v0.5.1

CtrlSpeak is a native Windows and Ubuntu/Linux speech-to-text client. Hold the **right Ctrl**
key to record, release it to transcribe, and CtrlSpeak inserts the result into
the active field. v0.5 adds signed in-application updates with a stable installed
filename. v0.5.1 adds an ordered output-language allowlist enforced by both the
embedded model and maintained remote API, while preserving the v0.4 Linux and
transcription-backend design:

- **Embedded / local** runs the bundled `faster-whisper` workflow and retains
  the existing model (`small` or `large-v3`) and device (`cpu` or `cuda`)
  settings.
- **Remote API** sends the recording to a configurable HTTP(S) base URL with an
  optional bearer token. It retains the API transcription ID and submits an
  edited final result to that same API endpoint.

The legacy **Client + Server** and **Client Only** roles remain inside the
embedded/local backend. Remote API mode is independent of those roles and does
not start discovery, a local server, or a model download.

## Output-language control

The control center's **Output languages** list applies to both transcription
backends. Select no languages for automatic, unrestricted Whisper detection;
select one to force that recognition language; or select up to five in priority
order. With several selected languages, an allowed detected language is used,
otherwise the first selected language is the fallback. **English only** and
**Clear (automatic)** provide quick choices.

This setting prevents CtrlSpeak from accepting a reported language outside the
configured list. It constrains speech recognition/decoding; it is not an
arbitrary translation feature. Backend and language changes are pinned for the
running process and take effect after restarting CtrlSpeak.

## Application updates

The standard packaged application is installed with one stable name:

- Windows: `CtrlSpeak.exe`
- Linux: `CtrlSpeak`

The tray menu shows the running version and contains **Check for updates**. The
same action is available in the control center's **Application updates** card.
Update checks and downloads run in the background and do not freeze recording,
the tray, or Tk.

CtrlSpeak accepts an update only when a signed stable manifest identifies the
exact standard product, operating system, x86-64 architecture, release tag, and
artifact. It verifies the Ed25519 manifest signature, declared byte length, and
SHA-256 before asking to restart. A copied external helper atomically replaces
the executable, waits for a health receipt from the new version, and restores
the previous verified executable automatically if startup fails. Settings,
models, CUDA files, logs, API credentials, and local corrections stay in the
per-user CtrlSpeak data directory and are not replaced.

A source checkout may check which release is published, but it cannot overwrite
itself with a release binary. Update source checkouts through Git. v0.4 and
earlier builds do not contain updater code, so v0.5.0 is the one final manual
replacement under the stable filename; v0.5.1 and later can use the GUI.

See `docs/UPDATING.md` for the end-user flow, release process, failure recovery,
and maintainer checklist.

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
  "allowed_output_languages": [],
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

When output languages are configured, the request includes an ordered
comma-separated `allowed_languages` multipart field such as `en` or `en,af`.
The maintained v0.5.1 server validates a maximum of five codes, forces one of
them, uses the first as a fallback, and refuses to return a reported language
outside the list. Omitting the field preserves automatic detection. The legacy
single `language` field is still accepted by the server.

The transcribe response must include non-empty `text`, `raw_text`, and `id`.
For a restricted request it must also report a `language` inside the configured
allowlist; the desktop client rejects an out-of-policy response.
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
export CTRLSPEAK_OUTPUT_LANGUAGES=en,af
python main.py
```

Backend settings are pinned at startup. Saving a backend, URL, token, or
feedback/language change in the management window requires a restart,
preventing a partially switched runtime.

The maintained CUDA API service, Ubuntu user-service setup, and server tests
are under `server/whisper_transcription`. See `docs/API.md` for authentication,
all routes, request/response examples, errors, and the v0.5.1 language-policy
contract. A running server publishes OpenAPI at `/openapi.json`, Swagger UI at
`/docs`, and ReDoc at `/redoc`.

## Embedded/local Whisper

Embedded mode preserves model download, storage, selection, CPU inference, and
the legacy client/server behavior. The default `small` model is downloaded on
first embedded launch and stored under the XDG CtrlSpeak directory. Local
approved edits create exact raw-transcript-to-final-text overrides in
`local-corrections.sqlite3`; similar transcripts are not broadly rewritten.

CPU is the safe default. On Linux, CtrlSpeak can select `cuda` when
`libcuda.so.1` reports an NVIDIA device and CTranslate2 can use the installed
CUDA/cuDNN runtime. CtrlSpeak v0.5 does **not** install Linux GPU drivers or
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
--version                     Print the source/runtime version and exit
```

Use `python main.py --help` for parser details. On Linux, remove a manually
installed executable, icon, and desktop file manually; CtrlSpeak does not run a
package manager or delete arbitrary installation paths.

## Windows/Linux packaging and launcher template

The maintained native build command is:

```bash
python -m utils.build_exe
```

It selects `packaging/CtrlSpeak_v0.5.spec` and produces the stable one-file
executable `dist/CtrlSpeak.exe` on Windows or `dist/CtrlSpeak` on Linux. The
native icon, updater verification key, and runtime assets are bundled; model
weights and system CUDA libraries remain external. PyInstaller does not
cross-compile the Linux artifact from Windows or vice versa.

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
python -m pytest -q tests/core/test_update_manager.py tests/core/test_update_helper.py
python -m pytest -q server/whisper_transcription/tests
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

Application updates trust only the hard-coded
`Wouter-Bekker-AI/CtrlSpeak` stable-release identity and the embedded Ed25519
public key. Repository identity, product, variant, and signing key are not
editable settings. HTTPS is required but does not replace signature, size, and
hash verification.

## License

MIT License. Copyright (c) 2025 CtrlSpeak contributors.
