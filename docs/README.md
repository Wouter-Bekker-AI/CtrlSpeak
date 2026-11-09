# CtrlSpeak documentation

Use these guides for the maintained release line. Versioned plans, incident
reports, and older deployment handoffs are records of what was planned or
verified at that version, not instructions to redeploy those versions.

| Need | Guide |
| --- | --- |
| Install, choose a backend or start dictating | [Project README](../README.md) |
| Shortcuts, mic feedback, Live safety, text recovery or corrections | [Desktop workflow](user_flow.md) |
| Integrate a client with the gateway or worker | [API contract](API.md) |
| Configure the service roles or external Nova STT helper | [Transcription service](../server/whisper_transcription/README.md) |
| Find the relevant code or understand compatibility constraints | [Architecture and code map](ARCHITECTURE.md) |
| Build locally and verify a candidate | [Build guide](../packaging/BUILDING.md) and [test gates](TESTING.md) |
| Update, publish, verify or roll back a release | [Update and release guide](UPDATING.md) |
| See what changed in each version | [Changelog](../CHANGELOG.md) |

The current maintenance release record is [0.8.6](V0.8.6_RELEASE.md).
The design rationale for the desktop is preserved in the
[Midnight Signal record](V0.7_MIDNIGHT_SIGNAL_RELEASE.md) and the
[Handy investigation](HANDY_PROJECT_LESSONS.md). Use the latest API and
update guides, not old release handoff commands, for ongoing operations.
