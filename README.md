# CtrlSpeak v0.8.6

CtrlSpeak records speech and inserts the transcript into the active text field.
Use it on Windows or an Ubuntu **X11/Xorg** desktop, with local Whisper or a
configured CtrlSpeak gateway. **0.8.6 polishes the Midnight Signal control
centre** without changing dictation modes or the API. See the
[release record](docs/V0.8.6_RELEASE.md).

## Get started

1. Download the Windows or Linux executable from the
   [latest signed release](https://github.com/Wouter-Bekker-AI/CtrlSpeak/releases/latest).
   Keep the installed filename `CtrlSpeak.exe` (Windows) or `CtrlSpeak` (Linux).
   If already installed, use **Check for updates** from the tray or control center.
2. Open the tray's **Open control centre**. Choose **Remote API** and enter the
   gateway base URL if one has been provided; use **Check gateway** to verify
   capabilities. Otherwise use **Embedded / local** and follow the first-run
   model setup. The defaults for local inference are CPU and Whisper small.
3. Focus a text field, hold **Right Ctrl**, speak, then release. The overlay
   shows recording and processing; the result is inserted when ready. If
   insertion fails, check the field and use **Copy last transcript**.

Linux requires a real X11 session for global shortcuts and insertion. Native
Wayland and headless desktop operation are not supported. Use **Ubuntu on Xorg**
at sign-in. A headless **server** does not have this desktop restriction.

## Dictation modes

| Mode | Start (defaults) | Finish | Output |
| --- | --- | --- | --- |
| Hold to Talk | Hold Right Ctrl | Release Right Ctrl | One transcript |
| Hands-free | Double-tap Right Ctrl | Tap Right Ctrl | One transcript |
| Live (Windows, opt-in) | Press Left + Right Ctrl, then release both | Tap Right Ctrl | Completed phrases as you speak |

Tap **Left Ctrl** alone to mute or resume Hands-free or Live. Press **Esc** to
cancel an active dictation; that cannot undo text already inserted. Open
**Guide** in the control center to watch the configured shortcuts without
recording. **Capture → Capture feedback** includes cue volume and the optional
**Show technical mic level (RMS dBFS)** setting. The default 0–100 readout is a
display scale: 0 is −60 dBFS or quieter, 50 is −30 dBFS, and 100 is 0 dBFS.

Live sends one request per phrase, not a streaming connection. Editing or
changing focus pauses delivery; uncertain text is held for **Copy pending** or
**Discard pending**. CtrlSpeak cannot identify every remote field inside
AnyDesk. See the [desktop workflow and safety guide](docs/user_flow.md).

## Transcription and corrections

- **Remote API** uses a configured base URL and optional gateway bearer token.
  It does not start a LAN server, discovery, or local model download. Backend,
  URL, token, route and output-language changes take effect after restart.
  An empty language list means automatic; select up to five in order to restrict
  recognition. This is not translation.
- The **gateway** routes to the Ubuntu GPU, caller-funded OpenAI GPT Transcribe,
  or CPU Whisper tiny. The GPU- and OpenAI-preferred routes try those providers
  in opposite orders, with tiny last. Provider-only options do not fall through.
  Routing shows the provider actually used and measured timing where available.
- Optional **S1 Mini cleanup** runs only on the Ubuntu GPU worker. If cleanup
  is not applied, use ordinary corrected text. **Preserve paragraphs and line
  breaks** is a separate desktop preference; leave it off for terminals and
  AnyDesk. Live always inserts single-line text and never submits Enter.
- **Submit correction…** in the tray adds a known-word rule to the authenticated
  gateway. Embedded mode instead supports exact local overrides. Whole-field
  edit feedback is optional and does not apply to individual Live phrases.
- An OpenAI key can be saved in Windows Credential Manager and is sent to the
  gateway only for individual requests. It is not persisted by the gateway or
  sent to the worker. Gateway bearer tokens saved in desktop settings are
  separate and stored in the user-local settings file. Use HTTPS or a trusted VPN.

See the [API contract](docs/API.md) and [server role and helper guide](server/whisper_transcription/README.md).
The external Nova/Hermes helper calls the gateway; it is not a transcription
server and does not require changes inside Hermes.

## Application updates

The updater checks the signed manifest, exact target, size and SHA-256. An
exact supported base can download a smaller differential patch; otherwise the
verified full download is used. **Restart and install** needs your confirmation.
The helper restores the previous executable if the new one fails its startup
health check. Settings, credentials, models, and corrections stay in their
existing locations. The stable updater only installs newer versions. A source
checkout updates through Git, not through the GUI.

See [Updating and releasing](docs/UPDATING.md) for exact steps, patch space
requirements, diagnostics, and rollback.

## Develop or build from source

Use Python 3.11 or 3.12 in a virtual environment. Install
`requirements.txt` to run from source, or `requirements-dev.txt` for tests and
static-analysis tools. See [Building CtrlSpeak](packaging/BUILDING.md) for
Windows and Linux system prerequisites.

```text
python main.py --help
python main.py --version
python main.py
python scripts/run_bounded_tests.py --timeout 180 -m core_headless
python -m utils.build_exe
```

The native build produces `dist/CtrlSpeak.exe` or `dist/CtrlSpeak`. It does
not install the app or deploy a service. Persistent app data lives below
`%APPDATA%\CtrlSpeak` on Windows or `${XDG_CONFIG_HOME:-~/.config}/CtrlSpeak`
on Linux. Build outputs and installed assets are not runtime data stores.
The legacy embedded LAN server and UDP discovery are for trusted networks
only; do not expose them publicly.

- [Code map and compatibility boundaries](docs/ARCHITECTURE.md)
- [Testing and physical acceptance](docs/TESTING.md)
- [Changelog and previous releases](CHANGELOG.md)
- [Project ground rules](docs/PROJECT_GROUND_RULES.md)

The maintained release source line is `stable`. Release tags identify exact
build inputs. Older branches and versioned plans/incident records are
historical; see the [documentation index](docs/README.md) for current guides.

## License

MIT License. Copyright (c) 2025 CtrlSpeak contributors. See
[third-party notices](packaging/THIRD_PARTY_NOTICES.md) for the bundled patch tool.
