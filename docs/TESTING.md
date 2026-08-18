# CtrlSpeak v0.7.2 Midnight Signal testing playbook

Run commands from the v0.7 repository root in a project-compatible Python
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
- tray-accessible, authenticated known-word correction submission with
  user/global scope and non-blocking UI dispatch.
- secure Windows Credential Manager persistence/forget behavior without
  writing the OpenAI key to application settings.
- the Midnight Signal phase state machine, sanitized provider snapshots,
  accurate dBFS calculation/smoothing, duration formatting, and rejection of
  arbitrary server error text.
- audio-cue volume/mute persistence and bounded gain application without
  changing pitch or retaining recording content.
- both provider-preference cascades, single-provider routes, quota fallthrough,
  and cached/circuit-broken fast failure for an offline Ubuntu GPU worker.
- ordered output-language validation, settings migration, API propagation, and
  refusal of out-of-policy API responses.
- exact-only local correction persistence.
- observer-only bare Enter handling with no suppression, replay, or duplicate.
- lazy Linux/Windows input routing and X11/Wayland capability reporting.
- preferred-xclip and no-xclip tkinter clipboard restoration/injection behavior,
  including Unicode/multiline selections and unavailable-display errors.
- Linux CUDA driver routing without Windows loader calls.
- v0.7 PyInstaller, native icons, desktop launcher, and AppStream metadata.
- preservation of historical specs while the standard helper selects v0.7.
- strict semantic versions and exact product/platform/architecture selection.
- Ed25519 manifest verification, immutable release URLs, size and SHA-256 checks.
- resumable bounded downloads, Range validation, cancellation, and oversize refusal.
- update-operation generation IDs and stale worker-event rejection.
- external replacement health confirmation and automatic rollback with dummy files.
- settings schema migration, atomic writes, per-field salvage, and backup.
- stable v0.7 Windows/Linux packaging and release-manifest tooling.

Focused commands:

```bash
python -m pytest -q tests/core/test_linux_platform.py
python -m pytest -q tests/core/test_linux_models.py
python -m pytest -q tests/core/test_packaging_v04_linux.py
python -m pytest -q tests/core/test_config_paths.py
python -m pytest -q tests/core/test_transcription_backend.py
python -m pytest -q tests/core/test_ui_state.py
python -m pytest -q tests/core/test_audio_cues.py
python -m pytest -q tests/core/test_midnight_visual_contract.py
python -m pytest -q tests/core/test_system_cli.py
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

## v0.7.2 panel-dismissal regression gates

Before v0.7.2 is tagged:

1. Run the deterministic tray and Midnight Signal visual-contract tests. They
   must verify the visible quick-panel and control-center hide actions, Escape,
   idempotent close, and the native tray's static Show/Hide toggle label.
2. Open the quick panel from the packaged tray. Confirm **Hide panel**, Escape,
   and **Show / hide quick panel** in the native menu each dismiss only the
   panel and leave the tray/hotkey process alive.
3. Move focus through every quick-panel control with keyboard and pointer.
   After dismissal, reopen it repeatedly and confirm exactly one responsive
   panel exists.
4. Open the full control center. Confirm **Hide to tray** remains visible in its
   persistent header on every page and is also present on System. Hide and
   reopen it without changing backend, route, microphone, or update state.
5. Record and transcribe before and after hiding both surfaces. Confirm that
   hide never invokes Quit, interrupts capture, or changes the v0.7.1 native
   audio-safety guarantees.
6. Repeat against the exact one-file Windows candidate and on Ubuntu/X11 where
   its tray backend is supported. Inspect logs for Tk cross-thread access,
   duplicate-window, or native tray errors.

## v0.7.1 native-audio regression gates

v0.7.0 passed the headless and packaged metadata checks but could still crash
inside `_portaudio` when right Ctrl started recording. The recorder and the new
recording-start cue could initialize separate PyAudio managers concurrently;
PortAudio's process-global Windows host state is not safe across those competing
lifecycles. The failure is a native access violation and therefore bypasses
Python exception handling and application logs.

Before v0.7.1 is tagged:

1. Run the deterministic audio-lifecycle tests. A guarded fake backend must
   reject any second PortAudio initialization or termination while capture is
   live. Cover start, normal release, cancellation while the recorder lingers,
   and shutdown.
2. Confirm cue delivery failure cannot prevent microphone capture and that
   cancelling a session cannot detach a still-live recorder or allow an old
   generation to clear newer state.
3. On Windows with a physical/default microphone, run at least 50 right-Ctrl
   press/release cycles with cues enabled, including rapid presses and
   cancellations. The process must remain alive and responsive.
4. Repeat that stress pass against the exact one-file `dist/CtrlSpeak.exe`, not
   only source mode. Inspect the matching Windows Application event-log window;
   there must be no Application Error/WER entry for `CtrlSpeak.exe`,
   `_portaudio*.pyd`, or exception `0xc0000005`.
5. Repeat with cues muted and with the output device unavailable. Recording
   must remain usable, and shutdown/startup must remove bounded stale recording
   files without retaining audio content.

For a read-only Windows visual pass that does not start a second hotkey listener
or model, use the source harness:

```powershell
python scripts/ui_smoke_midnight_signal.py --page Capture --duration 120
python scripts/ui_smoke_midnight_signal.py --page Routing --duration 120
python scripts/ui_smoke_midnight_signal.py --page Corrections --duration 120
python scripts/ui_smoke_midnight_signal.py --overlay recording --duration 30
python scripts/ui_smoke_midnight_signal.py --overlay processing --duration 30
python scripts/ui_smoke_midnight_signal.py --overlay success --duration 30
python scripts/ui_smoke_midnight_signal.py --flyout --duration 30
```

The harness reads configured gateway capabilities/corrections but neither
records audio nor transcribes. Its sample provider/timing data is confined to
explicit overlay render states and is never used by the real application.

## Physical Windows Midnight Signal acceptance

Run these checks against both source and the exact packaged
`dist/CtrlSpeak.exe` at 100%, 125%, 150%, and 200% display scaling where
available:

1. Open CtrlSpeak from the stable filename. Confirm the tray and control center
   show 0.7.2, render the graphite Midnight Signal hierarchy cleanly, remain
   readable at the supported scales, and expose visible keyboard focus.
2. Left-click the tray icon and verify the branded quick surface; right-click
   and verify the dependable native fallback menu. Exercise Open control
   centre, Copy last transcript, Submit correction, gateway refresh, Check for
   updates, mute/volume, and Quit. Disabled actions must look and behave disabled.
3. Hold right Ctrl on each connected monitor. Confirm the slim recording capsule
   appears on the active monitor without taking focus, its timer advances, the
   waveform/dBFS meter reacts to the microphone, and silence settles at the
   documented floor rather than displaying fake activity. Recording-start
   feedback must not crash, stall, or race the input stream.
4. Release right Ctrl. Confirm the capsule transitions to the elongated
   transcribing animation. It must not show providers arranged around the
   microphone or claim a provider before a result arrives.
5. Complete a direct GPU request and a controlled fallback request. Confirm the
   success state names only the provider returned by `provider_used`; the route
   view orders real attempts and labels probe, attempt, inference, and total
   routing durations separately. Missing values show unavailable, never sample
   numbers.
6. Exercise recording, processing, success, cancelled, microphone error,
   provider error, and exhausted-route states. Confirm overlays dismiss
   predictably and do not leave the listener, sound, or WAV running.
   Start a second dictation during the prior terminal acknowledgement and
   confirm the old dismissal timer cannot close the new capsule. Cancel while
   a recorder is flushing and confirm a new generation remains blocked until
   that recorder really exits.
7. Adjust cue volume, mute, and reduced-motion preferences; restart and verify
   persistence. Listen through headphones at a conservative system level and
   confirm the short finite processing cue has no harsh full-scale attack,
   discontinuity, pitch shift, loop, or overlapping orphan playback.
8. Navigate the control center without a mouse. Confirm logical focus order,
   readable state text independent of colour, no clipped labels, Escape/close
   behaviour, and that reduced motion removes nonessential continuous motion.
9. Confirm Copy diagnostics, logs, UI state snapshots, and tray detail contain
   no transcript text, audio, API key, bearer token, correction phrase, header,
   clipboard data, or raw server exception.
10. Quit during a deliberately blocked provider request. Shutdown must return
    after its bounded wait, attempt exact-path WAV cleanup, and remove any file
    that remained locked when CtrlSpeak next starts under its single-instance
    lock.

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
2. Start CtrlSpeak and verify the Midnight Signal management window and tray
   surfaces. Confirm
   **Open control centre** reopens/raises the single management window,
   **Submit correction…** creates an immediately visible gateway rule, and
   **Quit** stops the listener/tray cleanly.
3. Select a real microphone, hold right Ctrl, speak, release, and confirm the
   recording/transcribing capsule behaves as documented and the WAV is removed
   from the XDG temp directory after processing.
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

1. Install v0.7.1 as `CtrlSpeak.exe` or `CtrlSpeak` and confirm the tray/control
   center show 0.7.1.
2. Publish the controlled signed v0.7.2 release with both required platform
   artifacts and the three metadata assets.
3. Check for the update from the GUI, inspect version/size, download, and confirm
   the UI remains responsive.
4. Restart and verify the same stable path now reports 0.7.2 while API URL/token,
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
appstreamcli validate --no-net packaging/linux/io.trueai.ctrlspeak.metainfo.xml
```

Also inspect the bundle's startup log at
`$XDG_CONFIG_HOME/CtrlSpeak/logs/ctrlspeak.log` (or
`~/.config/CtrlSpeak/logs/ctrlspeak.log`) for missing shared libraries/backends.

## Legacy workstation automation

`python main.py --automation-flow` remains the v0.3 provisioned Windows
workstation harness. Its SendInput and bundled CUDA-wheel stages are not a Linux
acceptance test. Do not use its host-level guidance on an Ubuntu system.

## Companion API tests

The maintained v0.7.2 role-aware service is in `server/whisper_transcription`. Its
headless suite uses a fake model and does not download CUDA/model assets, bind a
network port, restart systemd, or change a firewall:

```bash
cd server/whisper_transcription
python -m pytest -q
python -m compileall app tests
```

The server tests cover the OpenAPI version/field, authentication boundary,
legacy single-language compatibility, ordered allowlist propagation, invalid
policy rejection, hard refusal of out-of-policy responses, corrections,
confirmed-text feedback, and explicit CPU runtime selection. The suite and
editable package metadata are checked on Python 3.11 and 3.12. A deployment
acceptance test must additionally use real audio against its explicitly
configured CUDA or CPU service and assert the response language occurs in the
requested allowlist.

The v0.7 server suite also verifies safe provider-health capability fields,
per-attempt and total-route durations, worker inference duration, fallback
error timing, and exclusion of credential content. Deployment acceptance must
compare those fields with observed provider order and must not infer or relabel
missing measurements.

## Production gateway and worker acceptance

After taking timestamped source/configuration/database rollback copies and
deploying one role at a time:

1. Confirm the dedicated `CtrlSpeak` gateway reports version 0.7.2 and role
   `gateway`. Record the unchanged Ubuntu worker version, confirm role `worker`,
   and verify protocol compatibility. Nova must not gain a CtrlSpeak listener.
2. Confirm the gateway retains all correction/audit rows, uses the existing
   authenticated identities, stores no OpenAI key, and listens only on its
   intended private WireGuard address.
3. Confirm capabilities publish the telemetry feature flags and bounded worker
   health values without secrets or transcript content.
4. Send real restricted-English audio through the gateway to the Ubuntu GPU.
   Assert non-empty corrected text, the requested language policy, the actual
   GPU provider, non-negative attempt/routing/inference timings, and no
   degradation.
5. With the worker deliberately unavailable, confirm the first health probe is
   bounded and the circuit makes later requests fail over quickly. Restore the
   worker and confirm the circuit closes after a successful bounded re-probe.
6. Confirm the Nova Hermes service remains active and can transcribe through
   the unchanged `/v1/transcribe` contract. Do not replace its host-specific
   command adapter during this deployment.
