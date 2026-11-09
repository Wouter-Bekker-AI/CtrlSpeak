# CtrlSpeak architecture and code map

The desktop, gateway, Ubuntu worker and Nova helper share a release version.
They are separate roles. The desktop GUI updater replaces only the installed
desktop executable; the release workflow coordinates server-side selectors.

## Desktop

| Area | Ownership |
| --- | --- |
| Startup and settings | [main.py](../main.py) parses maintenance arguments and pins backend config; [config_paths.py](../utils/config_paths.py) owns per-user paths, settings migration, and logging. [credential_store.py](../utils/credential_store.py) handles the native Windows vault. |
| Capture lifecycle | [system.py](../utils/system.py) owns the listener, PortAudio arbitration, recording/transcription lifecycle, tray and shutdown. [hotkeys.py](../utils/hotkeys.py) adapts platform key events. |
| Dictation modes | [dictation_modes.py](../utils/dictation_modes.py) recognizes gestures. [dictation_integration.py](../utils/dictation_integration.py) wires the controller to desktop owners. [dictation_runtime.py](../utils/dictation_runtime.py) owns continuous-session audio, queued results, mute and delivery guards. |
| Presentation | [gui.py](../utils/gui.py) owns Tk lifecycle, shared settings/update controllers and corrections dialog. [midnight_signal_ui.py](../utils/midnight_signal_ui.py) supplies the visible control center/flyout, [midnight_overlay.py](../utils/midnight_overlay.py) the recording capsule, and [ui_state.py](../utils/ui_state.py) the display model. The Guide is a simulation separate from input handling. |
| Audio feedback | [audio_cues.py](../utils/audio_cues.py) synthesizes bounded finite cues; playback goes through the native sink. PortAudio remains the capture and input-enumeration path, not the processing cue player. |
| Transcription | [transcription_backend.py](../utils/transcription_backend.py) selects embedded or remote at runtime. [models.py](../utils/models.py) owns downloaded local Whisper models/CUDA. [cancellable_http.py](../utils/cancellable_http.py) handles cancellable remote requests; [recording_recovery.py](../utils/recording_recovery.py) bounds explicit retry audio. |
| Text delivery | [dictation_text.py](../utils/dictation_text.py) prepares safe insertion text; [winio.py](../utils/winio.py) lazily selects the [Windows](../utils/windows_input.py) or [X11](../utils/linux_input.py) adapter. [feedback_capture.py](../utils/feedback_capture.py) observes user edits; [transcript_audit.py](../utils/transcript_audit.py) is private and may contain text. |
| Signed updates | [update_manager.py](../utils/update_manager.py) verifies manifests/downloads; [binary_delta.py](../utils/binary_delta.py) reconstructs exact targets; [update_helper.py](../utils/update_helper.py) owns restart, health and rollback. |

Midnight Signal is the only control-centre widget tree. Its settings and
update controllers bind to those visible controls directly. Embedded LAN
discovery and the local model path are separate from the Remote API gateway;
removing the old UI did not remove those runtime modes.
Tk changes must run on the owner thread; preserve the management queue and
generation checks when touching lifecycle code.

## Remote transcription

- [Service entry point](../server/whisper_transcription/app/main.py) validates
  and dispatches API requests. The gateway owns [auth](../server/whisper_transcription/app/auth.py),
  [routing](../server/whisper_transcription/app/providers.py),
  [admission](../server/whisper_transcription/app/admission.py), and the
  [correction store](../server/whisper_transcription/app/corrections.py).
- The Ubuntu worker handles CUDA inference and optional
  [S1 normalization](../server/whisper_transcription/app/normalization.py).
  S1 is not gateway CPU inference and cannot run on OpenAI/Tiny results.
- [Audio transport](../server/whisper_transcription/app/audio_transport.py)
  is shared with the desktop and external helper. FLAC preserves PCM; the
  gateway forwards compressed bytes where supported. Keep WAV compatibility.
- [The Nova command helper](../server/whisper_transcription/scripts/hermes-stt-api)
  uses a separate environment and calls the gateway. It is not part of Hermes
  source and does not own Telegram echo behavior.

Do not remove old API fields, SQLite migrations, exact overrides, auth modes,
LAN compatibility, or full-download/rollback paths on the strength of a static
usage count. Those have consumers outside the current desktop code. Maintain
the [API contract](API.md) and [ground rules](PROJECT_GROUND_RULES.md).

## Release and verification

[The standard spec](../packaging/CtrlSpeak_v0.7.spec) has a historical filename
but builds the current version. The [test workflow](../.github/workflows/test.yml)
runs Windows and Linux; [the release workflow](../.github/workflows/release.yml)
tests native builds, verifies signed assets and deploys reviewed Git revisions.
Deployment tooling lives in [deploy_fleet.py](../scripts/deploy_fleet.py) and
[deploy_host.py](../scripts/deploy_host.py). Stable tags, not a mutable branch
checkout, identify deployed code. Secrets, models, settings, installed binaries
and production SQLite are not source files.
