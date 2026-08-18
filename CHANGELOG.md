# Changelog

CtrlSpeak follows semantic versioning. Published binaries are immutable; a
consumed release is corrected by a newer patch rather than by moving its tag or
replacing its assets.

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
`docs/V0.7.1_HOTFIX_RELEASE.md`.
