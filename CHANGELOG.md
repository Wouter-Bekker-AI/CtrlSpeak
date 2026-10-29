# Changelog

CtrlSpeak follows semantic versioning. Published binaries are immutable; a
consumed release is corrected by a newer patch rather than by moving its tag or
replacing its assets.

## 0.8.2 — 2026-10-29

- Add signed VCDIFF executable patches, exact base-version/size/SHA-256
  selection, bounded local reconstruction, full target verification and a
  one-time full-download fallback. The running executable is never patched.
- Show selected download method/size, separate reconstruction progress and
  persisted redacted transfer receipts. Retain cancellation, resumable
  downloads, startup-health rollback and explicit signed 0.7.7 rollback.
- Publish and roundtrip-test direct patches from up to three recent
  patch-aware releases. Preserve schema-1 full assets for older updaters.
- Bundle hash-pinned Xdelta 3.2.2 and its Apache-2.0 license; never fetch a
  decoder during an installed-app update. Bound subprocess time/output and
  prevent overlapping updater workers.
- Check temporary free space and bound completed-update payload caches while
  retaining audit metadata, recent rollback payloads and all recovery states.
- Bootstrap requires a full 0.8.2 install. The proposed animated mode guide
  belongs to 0.8.3 and is not implemented in this release.

See [verification and deployment status](docs/V0.8.2_RELEASE.md).

## 0.8.1 — 2026-10-29

- Normalize generic pynput `Key.ctrl` events to Left Ctrl, fixing that event
  path for the two-Control Live shortcut and isolated Left Ctrl mute/resume.
  Retain explicit Right Ctrl identity and editing-chord protections.
- Place active Live's capsule at the top centre of the active monitor's work
  area, leaving bottom-aligned terminal/CLI input visible. Keep its position
  through mute, editing, processing and completion; restore the established
  bottom position for the next Hold/Hands-free capture, including reused windows.
- Add controller/monitor, actual Windows pynput-enum and high-DPI/secondary-
  monitor placement regressions. Preserve signed rollback to 0.7.7 and all
  existing provider/API/cleanup behavior.

Build, publication and deployment evidence is recorded in the
[0.8.1 release guide](docs/V0.8.1_RELEASE.md).

## 0.8.0 — 2026-10-29

- Keep Hold to Talk and add Hands-free Dictation (double-tap primary key) and
  opt-in Windows Live Dictation (primary + secondary keys). Allow distinct
  Left/Right Ctrl or F8/F9/F10 assignments with updated on-screen hints.
- Add isolated secondary-key mute/resume, primary-key finish and explicit
  cancellation, with persistent mode/mute/editing/processing indicators.
- Send Live speech in bounded phrases after 650 ms silence or 8 seconds;
  preserve lossless audio transport and the configured provider/S1 policy.
- Pause Live around physical editing and focus changes. Hold stale or uncertain
  results for verified Copy/Discard rather than automatically replaying them.
  Never rewrite committed text or submit an Enter on a user's behalf.
- Flatten formatting only in Live and add a trusted ordinary-space separator;
  retain Hold/Hands-free paragraph preferences and API/Telegram formatting.
- Bound session/audio/queue/held-text resources, discard muted audio, and report
  backlog, partial insertion and failed requests without automatic retries.
- Add signed, user-requested rollback to 0.7.7 with settings and native
  credential-vault preservation. Keep coordinated gateway/worker/external-helper
  deployment; Windows installation remains a user action.

Published after signed Windows/Linux package checks and coordinated backend
verification. Automated evidence and remaining interactive acceptance checks
are recorded in [the 0.8.0 release guide](docs/V0.8.0_RELEASE.md).

## 0.7.7 — 2026-10-22

- Replace per-character Windows injection delays with bounded, surrogate-safe
  batches; stop on cancellation/focus change and never replay partial delivery.
- Add cancellable remote transcription, separate connection/read/total limits,
  typed actionable errors and explicit processing-stage telemetry.
- Offer manual Retry/Discard for one short-lived, memory-only failed recording;
  never automatically resend audio or retype a possibly inserted transcript.
- Bound worker/gateway admission, expose worker busy/stalled state and route
  busy workers to the next permitted provider without a long offline cooldown.
- Add correlated request/stage diagnostics and low-disk resource warnings;
  avoid sending provider-independent invalid audio through every fallback.
- Preserve lossless transport, correction rules, S1 selection/formatting and
  coordinated signed release deployment. Windows clients remain user-updated.

## 0.7.6 — 2026-09-19

- Separate S1 cleanup from desktop formatting: optional paragraph checkbox,
  visible only with cleanup enabled, safe single-line default and immediate
  saved formatting policy at injection. No API/Telegram formatting change.
- Block command control characters, normalize whitespace, prevent trailing
  submit characters, and encode non-BMP Windows input as proper UTF-16 pairs.
- Retain S1 and final normalized text in gateway audit records; add a bounded,
  escaped private desktop audit of selected and insertion text and outcomes.
- Coordinate gateway, Ubuntu worker/desktop and external Nova helper through
  the signed release pipeline; Windows clients update through the GUI.

## 0.7.5 — 2026-09-18

- Add lossless FLAC transport, compressed pass-through and bounded decoding.
- Add signed, staged fleet deployment with rollback and live canaries before
  publishing the Windows/Linux update. Keep Hermes independent of its helper.

## 0.7.4 — 2026-09-15

- Add opt-in S1-mini by Superwhisper cleanup to the Ubuntu CUDA worker, with a
  saved desktop checkbox, explicit `cleanup` API field and validated selection.
- Remove gateway CPU cleanup and its delay for clients that do not use it.
- Preserve ASR on cleanup failure: bounded input/output, hard total deadline,
  no waiting queue and a cooldown. OpenAI/Tiny bypass cleanup entirely.
- Reconcile correction-preview handling into versioned source; preserve raw and
  ordinary corrected fields, exact overrides, identities and legacy clients.

## 0.7.3 — 2026-08-18

### Fixed

- Replace the legacy correction-submission popup with a responsive Midnight
  Signal surface whose form body can scroll while status and actions remain
  visible at supported display scales and on short work areas.
- Use the real CtrlSpeak microphone artwork in both recording and processing
  overlays, preserving the same brand identity across capture phases.
- Repair **Copy last transcript** on 64-bit Windows by declaring pointer-sized
  Win32 clipboard APIs, providing a valid clipboard owner, serializing access,
  retrying bounded contention, and verifying the staged Unicode text.

### Release engineering

- Synchronize desktop, service, Windows PE, AppStream, API documentation, and
  signed-release version gates at 0.7.3.
- Add regression coverage for correction-dialog reachability, packaged brand
  artwork, tray-thread clipboard dispatch, clipboard ownership, and read-back
  failure reporting.

## 0.7.2 — 2026-08-18

### Fixed

- Add an always-visible **Hide panel** action to the Midnight Signal quick
  panel, plus Escape and window-close dismissal.
- Add **Hide to tray** to the persistent control-center header and System page
  so the management window never traps the user in an always-visible surface.
- Rename the native tray action to the truthful static label **Show / hide
  quick panel**, avoiding unsafe cross-thread Tk or native-menu state reads.
- Keep hide/close idempotent and separate from **Quit CtrlSpeak**, preserving
  recording, transcription, configuration, and tray availability while UI
  surfaces are hidden.

### Release engineering

- Synchronize desktop, service, Windows PE, AppStream, API documentation, and
  signed-release version gates at 0.7.2.
- Add deterministic visibility, repeated-close, and static tray-toggle
  regression coverage for the dismissal hotfix.

## 0.7.1 — 2026-08-18

### Fixed

- Prevent competing PortAudio lifecycles between Windows microphone capture
  and asynchronous recording/cancellation feedback, which could crash the
  packaged v0.7.0 application on right-Ctrl press with a native `_portaudio`
  access violation.
- Keep cue playback failure isolated from recording and retain bounded
  recorder shutdown and temporary-audio cleanup.
- Correct control-center, tray-flyout, recording, processing, and terminal-state
  visual fidelity against the approved Midnight Signal direction.

### Release engineering

- Synchronize desktop, service, Windows PE, AppStream, API documentation, and
  signed-release version gates at 0.7.1.
- Verify packaged Windows file/product versions against the release tag.
- Require real-device and frozen-executable recording stress plus a Windows
  event-log check; headless audio mocks alone are not release acceptance.

## 0.7.0 — 2026-08-18

- Introduced the Midnight Signal control center, tray flyout, recording and
  processing surfaces, accessible state presentation, configurable audio cues,
  and truthful provider/routing telemetry.
- Preserved the v0.6 gateway, GPU-worker, BYOK, correction, language-policy,
  and signed self-update contracts.

Detailed release contracts are in
`docs/V0.7_MIDNIGHT_SIGNAL_RELEASE.md` and
`docs/V0.7.1_HOTFIX_RELEASE.md`, with the current dismissal hotfix in
`docs/V0.7.2_HOTFIX_RELEASE.md` and the current desktop-fidelity hotfix in
`docs/V0.7.3_HOTFIX_RELEASE.md`.
