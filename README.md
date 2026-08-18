# CtrlSpeak v0.7.3

CtrlSpeak is a native Windows and Ubuntu/Linux speech-to-text client. Hold the **right Ctrl**
key to record, release it to transcribe, and CtrlSpeak inserts the result into
the active field. v0.5 adds signed in-application updates with a stable installed
filename. v0.5.1 adds an ordered output-language allowlist enforced by both the
embedded model and maintained remote API, while preserving the v0.4 Linux and
transcription-backend design. v0.6 adds capability-aware gateway routing and a
dedicated GPU worker role. v0.7 introduces the **Midnight Signal** experience:
a focused dark control center, a compact hold-to-record capsule, a calm
elliptical processing animation, a more useful tray surface, truthful provider
timings, and configurable peak-bounded feedback sounds.

- **Embedded / local** runs the bundled `faster-whisper` workflow and retains
  the existing model (`small` or `large-v3`) and device (`cpu` or `cuda`)
  settings.
- **Remote API** sends the recording to a configurable HTTP(S) base URL with an
  optional bearer token. It retains the API transcription ID and submits an
  edited final result to that same API endpoint.

The legacy **Client + Server** and **Client Only** roles remain inside the
embedded/local backend. Remote API mode is independent of those roles and does
not start discovery, a local server, or a model download.

## v0.7 Midnight Signal interface

Midnight Signal combines the clean visual restraint of the original Midnight
Glass concept with Signal Studio's operational detail. The interface uses a
graphite surface, restrained cyan/green state accents, clear typography, and
progressive disclosure. It does not arrange providers around a microphone or
pretend to know which provider will win before the gateway responds.

The recording overlay is a slim capsule with a live timer, waveform/level
feedback, and a correctly labelled dBFS reading. Releasing right Ctrl moves the
same surface into a calm transcribing state with an elongated elliptical motion
treatment. Success names the provider actually reported by the result; a
fallback path is shown only after the gateway returns measured attempts.

The tray and control center expose the useful detail without turning every
transcription into a diagnostic task:

- current microphone and live input level;
- selected route and provider readiness;
- measured probe, routing, attempt, and inference durations when supplied;
- fallback/degraded state and safe failure categories;
- **Copy last transcript**, **Submit correction…**, audio-cue controls, gateway
  check, update status, settings, and Quit; and
- the existing corrections, secure OpenAI credential, language, model, device,
  feedback, and signed-update controls in a cleaner information hierarchy.

Telemetry is intentionally truthful. Missing or unsupported measurements show
an em dash or an explanatory unavailable state; client wall-clock time is not
labelled as server inference, and illustrative concept values never appear as
live data. UI snapshots retain no transcript text, audio, bearer credentials,
or OpenAI key. See `docs/V0.7_MIDNIGHT_SIGNAL_RELEASE.md` for the approved
experience baseline, `docs/V0.7.1_HOTFIX_RELEASE.md` for the recording-safety
hotfix, `docs/V0.7.2_HOTFIX_RELEASE.md` for dismissal and tray visibility, and
`docs/V0.7.3_HOTFIX_RELEASE.md` for the current correction-dialog, overlay-brand,
and Windows clipboard hotfix contract.

## v0.7.3 desktop fidelity and clipboard hotfix

v0.7.3 brings tray-launched correction submission into the Midnight Signal
interface. Its responsive, scrollable body adapts to active-monitor work-area
and display scaling while keeping status, **Hide**, and **Submit correction**
actions outside the scrolling region. Keyboard users can dismiss with Escape
and submit with Ctrl+Enter or Alt+S.

The recording and processing capsules now use the packaged CtrlSpeak microphone
artwork rather than a generic drawn symbol. The asset is selected through the
same platform-aware packaging path used by the application icon and retained at
the rendered size so it remains present throughout animation.

On 64-bit Windows, **Copy last transcript** now opens the clipboard with a valid
owner window and pointer-sized Win32 handle declarations, serializes and retries
bounded clipboard access, and verifies the Unicode text after staging it. A
failed write is reported as a failure instead of showing a false success.

## v0.7.2 quick-panel dismissal hotfix

v0.7.2 makes every persistent Midnight Signal surface visibly dismissible
without quitting CtrlSpeak. The compact quick panel now includes **Hide panel**,
closes with Escape, and remains available from the truthful native tray action
**Show / hide quick panel**. The full control center exposes **Hide to tray** in
both its persistent header and System page. Hiding either surface leaves the hotkey, transcription,
gateway connection, and tray process running; **Quit CtrlSpeak** remains the
only application-exit action.

The tray label is deliberately static: neither the Tk thread nor the native
tray thread reads or refreshes the other UI surface's state. Repeated hide calls
are idempotent, and reopening creates one usable panel without orphaned polling
jobs or duplicate controllers.

## v0.7.1 recording and visual-fidelity hotfix

v0.7.1 removes a native Windows crash discovered after v0.7.0 was packaged.
The recording-start and cancellation feedback paths can no longer create a
second competing PortAudio lifecycle while microphone capture is starting or
stopping. A cue-backend failure remains non-fatal to recording, and CtrlSpeak
continues to bound shutdown and attempt cleanup of only that session's exact
temporary recording. If Windows still holds the file, the next locked startup
also performs bounded stale-recording cleanup.

This patch also corrects the visual fidelity of the control center, tray
flyout, and recording/processing surfaces against the approved Midnight Signal
direction. The corrections retain readable state text, restrained animation,
truthful provider telemetry, reduced-motion behaviour, keyboard access, and
on-screen placement at supported Windows scaling factors.

## v0.6.2 routing and credential patch

The gateway now publishes separate `ubuntu-gpu-preferred` and
`openai-preferred` cascades, plus `ubuntu-gpu-only`, `openai-only`, and
`gateway-tiny-only`. Cascades continue to the next provider when an upstream
is offline, rejects a key, or has no available credit; the failed attempt and
reason remain visible in response metadata. Single-provider routes return that
provider's exact error without falling through.

The Ubuntu worker uses a 500-ms health probe with a 350-ms connection ceiling.
A failed probe opens a 30-second circuit, so subsequent recordings bypass the
offline worker immediately instead of repeating a slow connection attempt.
Healthy results are cached briefly and a successful transcription closes the
circuit.

On Windows, **Remember securely on this computer** saves the OpenAI key in the
current user's Windows Credential Manager vault. CtrlSpeak loads it on future
starts and provides **Forget key** to remove it. The key is never written to
`settings.json`, stored by the gateway, sent to the Ubuntu worker, or included
in logs. Platforms without a supported native vault remain session-only.

## v0.6.1 correction submission patch

The tray now includes **Submit correction…**. It opens a small authenticated
form for entering the phrase CtrlSpeak currently produces and the replacement
it should return. By default, the new rule belongs to the current gateway
identity. Administrators can choose to make it global for every gateway user.
Successful rules become active immediately; secrets and correction text are
not written to application logs.

## v0.6 gateway and provider routing

The desktop now checks `GET /v1/capabilities` before treating a remote endpoint
as its gateway. The control center's **Check gateway** button shows available
providers and populates the published routing strategies. A worker-only
endpoint is rejected as a desktop backend.

The production layout separates responsibility:

- the dedicated OpenStack instance named `CtrlSpeak` is the authenticated
  gateway and owns routing, identity-scoped known words/corrections, feedback,
  and audit records;
- the local Ubuntu GPU machine is a raw `large-v3-turbo` CUDA inference worker
  reachable by the gateway over WireGuard;
- OpenAI `gpt-transcribe` is an optional paid provider using the caller's own
  key, followed by the gateway's lazy CPU `tiny` emergency fallback; and
- the existing Nova instance remains the WireGuard transport hub only. It no
  longer runs the CtrlSpeak application gateway.

The default `ubuntu-gpu-preferred` cascade is Ubuntu GPU → OpenAI → gateway
tiny. `openai-preferred` reverses the first two providers. Cascading routes
continue after unavailable GPU, invalid/missing OpenAI key, or exhausted
OpenAI credit; responses state which provider ran, which attempts failed, and
whether the result is degraded. The `ubuntu-gpu-only` and `openai-only` routes
do not fall through.

The OpenAI key entered in the control center may be stored in Windows
Credential Manager for the current Windows user. It is never written to
`settings.json`, packaged into the executable, stored by the gateway, or sent
to the Ubuntu worker. It is attached only to individual gateway transcription
requests. The gateway must still be trusted because its process handles the
request transiently.

## v0.5.3 quality-of-life patch

The tray now includes **Copy last transcript**. CtrlSpeak keeps exactly one
successful transcription in memory, records it before attempting active-field
insertion, and can therefore recover the text when focus changed or a target
application rejected the paste. The item is disabled until a result exists;
copying gives explicit success or failure feedback. Transcript content is never
written to settings, logs, the correction database, or any other persistent
history, and is discarded when CtrlSpeak exits.

The looping processing chime now has a transparent -6 dBFS peak ceiling. The
bundled WAV previously reached almost full scale during its opening attack;
v0.5.3 attenuates that clip uniformly at load time so it is less harsh through
loud headphones without hard clipping or changing its pitch.

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

v0.5.2 fixes post-update HTTPS checks on one-file builds. A newly launched
version now replaces certificate paths inherited from the previous temporary
PyInstaller extraction, and updater subprocesses no longer pass those temporary
paths forward. Copied update diagnostics also include the redacted status
message alongside the finite error category.

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
models, CUDA files, logs, native credential-vault entries, and local
corrections are not replaced.

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
  "provider_strategy": "server-default",
  "device_preference": "cpu",
  "model_name": "small"
}
```

The API URL is a complete configurable `http://` or `https://` base URL. It may
point to loopback, a LAN/VPN host, or an HTTPS service. No current LAN address
is hardcoded. API mode calls:

- `GET <base-url>/v1/capabilities` to verify the gateway and list routes.
- `POST <base-url>/v1/transcribe` with multipart WAV audio.
- `POST <base-url>/v1/transcriptions/{id}/feedback` with the confirmed final
  text, capture method, and client audit metadata.

When output languages are configured, the request includes an ordered
comma-separated `allowed_languages` multipart field such as `en` or `en,af`.
The maintained v0.7.3 gateway validates a maximum of five codes, forces one of
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
export CTRLSPEAK_PROVIDER_STRATEGY=ubuntu-gpu-preferred
python main.py
```

Backend settings are pinned at startup. Saving a backend, URL, token, or
feedback/language change in the management window requires a restart,
preventing a partially switched runtime.

The maintained configurable API service, Ubuntu user-service setup, and server
tests are under `server/whisper_transcription`. See `docs/API.md` for authentication,
all routes, request/response examples, errors, and the current language-policy
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

It selects `packaging/CtrlSpeak_v0.7.spec` and produces the stable one-file
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
python -m pytest -q tests/core/test_ui_state.py tests/core/test_audio_cues.py
python -m compileall .
git diff --check
cd server/whisper_transcription
python -m pytest -q tests
cd ../..
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
