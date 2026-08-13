# Lessons from the Handy speech-to-text project

Status: design reference for current and future CtrlSpeak work

Review date: 2026-08-14

Project reviewed: [cjpais/Handy](https://github.com/cjpais/Handy)

Source snapshot: [`549cbde3ebb72459f7f7230783931a45222018a1`](https://github.com/cjpais/Handy/commit/549cbde3ebb72459f7f7230783931a45222018a1)

Release examined: [`v0.9.5`](https://github.com/cjpais/Handy/releases/tag/v0.9.5)
Licence at the reviewed snapshot: MIT

## 1. Why this document exists

Handy is a mature, cross-platform, offline speech-to-text desktop application.
Although it uses Tauri, React, TypeScript, and Rust rather than CtrlSpeak's
Python, Tk, and PyInstaller stack, it solves many of the same product and
reliability problems:

- a tray-first desktop experience;
- recording and transcription state that must remain responsive;
- downloadable models and large-file progress;
- packaged application updates;
- settings that survive upgrades;
- platform-specific capabilities and packaging; and
- unobtrusive communication of new features.

This document preserves the reusable ideas found during the source-level
review. It is intentionally separate from
`docs/V0.5_SELF_UPDATE_PLAN.md`: that plan specifies what v0.5 will build,
while this file is a longer-lived idea bank for later CtrlSpeak releases.

This is an engineering study, not a proposal to copy Handy's code. Any adopted
idea must be reimplemented for CtrlSpeak and checked against the Handy licence,
CtrlSpeak's architecture, and CtrlSpeak's privacy and security requirements.

## 2. High-level architecture observed

Handy separates its application into three broad layers:

1. A React/TypeScript UI for settings, onboarding, update status, model
   selection, recording history, and the recording overlay.
2. A Rust/Tauri application layer that owns the tray, native windows, commands,
   application lifecycle, settings persistence, and updater integration.
3. Rust managers for audio capture, transcription, models, history, and large
   downloads.

The most reusable architectural lesson is the ownership boundary. Native and
long-running operations are not allowed to leak into arbitrary UI components.
The UI receives explicit events and invokes a small command surface. CtrlSpeak
should aim for the same separation even though it remains a Python application:

- Tk and tray code render state and collect intent;
- a coordinator owns each long-running operation;
- services implement networking, verification, settings, and transcription;
- worker threads report immutable events rather than directly manipulating Tk;
  and
- platform-specific behavior is isolated behind narrow adapters.

## 3. Updater lessons

### 3.1 What Handy does well

Handy's installed build uses Tauri's updater plugin. Its packaged configuration
contains:

- a public update-signing key embedded in the application;
- a stable `latest.json` endpoint on GitHub Releases;
- per-platform and per-architecture manifest entries;
- signed update artifacts; and
- immutable, version-tagged asset URLs.

The UI distinguishes the important states instead of presenting update work as
one blocking action:

- checking;
- up to date;
- update available;
- preparing;
- downloading with percentage progress;
- installing; and
- failed.

It supports both automatic checks, when enabled in settings, and a manual tray
action. Manual checks give positive feedback when the application is already up
to date, whereas background checks can remain quiet. That distinction avoids
needless notifications without making a user-initiated action feel broken.

The tray menu includes a non-clickable version label and a dedicated **Check
for updates** item. This is directly relevant to CtrlSpeak: version visibility
and update discovery should not require knowing where the executable came from.

### 3.2 Installed and portable builds are different products operationally

Handy explicitly detects a portable mode. A marker beside the executable moves
settings, models, recordings, databases, and logs into a nearby `Data`
directory. Portable mode also changes update behavior: a running portable
Windows executable cannot simply replace itself, so Handy directs the user to
the correct installer instead of pretending that the installed updater path
will work.

The useful lesson is not the exact marker-file scheme. It is that runtime mode
must be classified before enabling update actions. CtrlSpeak should distinguish:

- a packaged executable that its helper can replace;
- a source checkout, which must not be overwritten by the binary updater;
- a packaged executable in an unwritable location;
- a Windows build versus a Linux build; and
- client-only, server-only, and combined application roles.

The v0.5 updater therefore needs explicit eligibility checks and clear messages.
It must never offer a successful-looking update flow that cannot safely finish.

### 3.3 Manifest URLs should be selected, not guessed

Handy's portable-installer resolver reads the matching platform entry from the
update manifest. It does not rebuild an asset URL from a naming convention.
This protects the client from repository renames and packaging-name changes.
It also ensures the selected URL points at a specific release tag rather than a
moving `latest` download.

CtrlSpeak should use the same principle:

- match an exact product, operating system, architecture, and package type;
- require an immutable tag-specific HTTPS URL;
- reject an absent or ambiguous target;
- validate the manifest before using any URL; and
- never choose the first `.exe` or `.bin` that happens to appear in a release.

### 3.4 What CtrlSpeak needs beyond Handy

Tauri owns much of Handy's replacement behavior. CtrlSpeak is a single-file
PyInstaller application and needs an application-specific external helper.
CtrlSpeak's stronger v0.5 contract should include:

- SHA-256 and declared-size verification in addition to the signed manifest;
- a same-directory candidate, current executable, and backup;
- atomic replacement where the operating system permits it;
- a transaction record in the application-data directory;
- a post-relaunch health receipt from the new application;
- a bounded health timeout; and
- automatic rollback and relaunch of the verified previous executable.

The key lesson is to copy Handy's clear product experience, not to assume its
framework-provided updater guarantees apply to CtrlSpeak.

## 4. Large-download engineering lessons

Handy's model downloader is particularly relevant because application binaries
and speech models are both large, failure-prone downloads. Its implementation
contains several practices worth retaining:

- partial files are distinct from verified final files;
- interrupted transfers can resume;
- a Range request is accepted only with a matching `206 Partial Content`
  response and correct `Content-Range` starting offset;
- a server that answers a Range request with `200 OK` causes a clean restart,
  not an append that would corrupt the file;
- `416 Range Not Satisfiable` is treated cautiously;
- finite header and mid-body stall timeouts prevent an operation from hanging
  forever;
- declared and observed file sizes are checked;
- the transfer is capped so a faulty or malicious server cannot fill the disk;
- progress events are throttled rather than flooding the UI;
- cancellation leaves only a resumable, untrusted partial;
- the final checksum is the trust anchor; and
- finalization occurs only after verification.

For CtrlSpeak, a partial download must additionally be bound to the exact signed
manifest entry. A partial from another version, channel, platform, architecture,
URL, size, or hash must not be reused.

The UI should show bytes, total size when known, percentage, and a meaningful
phase label. It should not display `100%` as installed while hashing,
replacement, or restart remains outstanding.

## 5. Concurrency and stale-event lessons

Handy's stores and managers reconcile backend state with frontend state and
clean up progress entries on completion, cancellation, failure, and unexpected
IPC errors. This highlights a common desktop-app problem: an older asynchronous
operation can finish after a newer operation starts and overwrite the new UI.

CtrlSpeak should use an explicit `UpdateCoordinator` with monotonically
increasing operation generations:

- every check or download gets a generation identifier;
- every worker event carries that identifier;
- cancelling or retrying advances the active generation;
- the UI ignores events from older generations; and
- only the coordinator can transition update state.

This pattern is also useful beyond updates. Future model downloads, server
health probes, transcription retries, device enumeration, and asynchronous
settings validation can use the same approach.

## 6. Settings and migration lessons

Handy has a typed settings model, explicit defaults, update commands, and
version-aware behavior. Its portable implementation also demonstrates that all
settings and data-path consumers should use one resolver rather than calculating
paths independently.

For CtrlSpeak, the durable version of this idea is:

- add a settings schema version;
- keep one authoritative settings path resolver;
- load JSON defensively;
- validate and normalize each field independently;
- salvage valid fields if one field is malformed;
- preserve unknown fields where forward compatibility benefits from it;
- back up the pre-migration settings file;
- write changes atomically through a temporary file and replacement; and
- never erase API endpoints, hotkeys, model choices, or other valid user
  preferences merely because one new updater field is invalid.

Updater-related settings should be modest. Sensible candidates are update-check
preference, channel, last successful check, dismissed release, last-seen What's
New version, and transaction-recovery state. Tokens and other secrets must never
enter updater logs or release manifests.

## 7. What's New and release communication

Handy bundles Markdown release notes and records the last version whose note the
user dismissed. A gate compares the running version with that value and opens a
one-time What's New modal after an upgrade. Users may turn the behavior off.

CtrlSpeak can reuse the product pattern with a simpler native UI:

- package a short, plain-text-safe note for each application release;
- show it only after the new executable has passed the health handshake;
- store the last-seen version only when the user dismisses it;
- offer an opt-out;
- keep release notes non-interactive unless a URL has been explicitly validated;
  and
- do not render remote HTML or unrestricted Markdown in Tk.

This turns updates into an understandable product event and makes newly added
controls discoverable.

## 8. Tray and low-friction workflow ideas

Handy's tray changes with application state. In the reviewed code it can expose
a version label, cancel an active recording/transcription, copy the last
transcript, select a downloaded model, unload a model, open settings, check for
updates, and quit.

Useful CtrlSpeak ideas include:

- always-visible version text in the tray menu;
- **Check for updates** in both the tray and management window;
- **Copy last transcript** as a fast recovery path when injection targets the
  wrong window or clipboard restoration fails;
- state-aware **Cancel** during recording/transcription;
- a concise backend indicator such as `Remote GPU: connected` or
  `Local model: ready`;
- model selection from the tray only after model lifecycle is centralized; and
- disabling actions whose prerequisites are unavailable instead of letting them
  fail later.

The menu should remain compact. Detailed networking, model, and diagnostic
controls belong in the management window.

## 9. Recording and transcription UX ideas

Handy demonstrates several broader interaction ideas that can improve CtrlSpeak
after the updater work is stable:

- a small recording overlay that provides immediate state feedback;
- a clear distinction between recording, transcribing, success, cancellation,
  and error;
- configurable start/stop sounds with a mute option;
- explicit accessibility/permission onboarding;
- audio-device selection and re-enumeration;
- a transcription history with retention controls;
- a reliable copy-last-transcript action;
- model download, verification, extraction, and activation as separate phases;
  and
- localization and right-to-left layout as first-class concerns.

CtrlSpeak should retain its quick global-hotkey workflow. Any overlay must be
optional, must not steal keyboard focus, and must not interfere with the target
application that will receive the transcript.

## 10. Model-management ideas

Handy models downloads as durable backend operations rather than ephemeral
button state. The frontend reconciles itself with backend truth on load and
listens for explicit progress, verification, extraction, completion,
cancellation, and failure events.

If CtrlSpeak later manages local models, it should adopt these ideas:

- a signed or otherwise trusted model catalog;
- explicit compatibility and disk-space information;
- download resume and cancellation;
- hash verification before a model becomes selectable;
- separate download, verify, extract, and load states;
- recovery when the UI restarts during a download;
- deletion that cannot remove the currently active model without a transition;
  and
- backend state as the source of truth.

For the current Windows-client/Ubuntu-GPU arrangement, a lighter equivalent
would show the remote server's health, loaded model, accelerator, and reported
server version without pretending the Windows client owns that model.

## 11. Release-engineering lessons

Handy builds multiple native targets and produces updater artifacts as part of
packaging. The useful release discipline for CtrlSpeak is:

- one source version drives application metadata, artifacts, manifest, and tag;
- build in a clean CI environment;
- test before packaging;
- build all supported platform/architecture targets;
- inspect the artifacts actually produced rather than assuming filenames;
- create a draft release first;
- generate checksums and the signed manifest from those exact bytes;
- smoke-test packaged executables;
- publish only after the whole required matrix succeeds; and
- make the manifest the final publication step so clients cannot discover a
  half-created release.

The installed executable should have a stable local name such as
`CtrlSpeak.exe`. Version numbers belong in application metadata, the tray,
About/management UI, Git tags, release assets, and manifests—not in the user's
day-to-day launch filename.

## 12. Privacy and security observations

Some Handy features need different choices in CtrlSpeak:

- Transcript history is useful but should be opt-in, bounded, and clearly
  deletable. CtrlSpeak should default to not persisting transcript content.
- Diagnostic logs must describe phases, versions, hashes, byte counts, and
  error categories without logging API tokens, audio, or transcript text.
- Any post-processing service or LLM integration must be separately consented
  to and must clearly disclose whether text leaves the machine.
- Custom-word replacement should use deterministic token/phrase boundaries,
  not fuzzy substitutions that can silently alter unrelated words.
- Remote release notes must be treated as untrusted text.
- Update URLs must come only from a verified manifest and an approved HTTPS
  origin.

## 13. Recommended adoption map

### Adopt in CtrlSpeak v0.5

- Stable local executable name.
- Version label in the tray and management window.
- Manual check in both places.
- Quiet optional background checking and explicit manual feedback.
- Exact signed platform/architecture manifest entries.
- Immutable release URLs.
- Clear check/download/verify/install/restart states.
- Resumable, bounded, hash-verified downloads.
- Operation generations and stale-callback rejection.
- Installed/source/runtime eligibility classification.
- Settings schema migration and per-field salvage.
- One-time What's New after verified update success.
- Draft-and-audit release workflow.
- External replacement helper with CtrlSpeak-specific health check and rollback.

### Good candidates after v0.5

- Copy last transcript from the tray.
- Optional non-focus-stealing recording overlay.
- Remote GPU server status and version display.
- Richer diagnostic/status screen.
- Configurable audio cues.
- Audio-device selection.
- Local model catalog and durable download manager.
- Opt-in transcription history with retention and delete controls.
- Permission/onboarding improvements for each operating system.
- Localization infrastructure.
- An explicit portable-data mode, if users need fully portable installs.

### Do not adopt without a separate privacy/security design

- Default transcript persistence.
- Transcript-content logging.
- Free-form cloud/LLM transcript rewriting.
- Fuzzy custom-word replacement.
- Installing from an unsigned or tag-only release.
- Guessing update assets from filenames.
- Silent forced updates.
- Updating source checkouts by overwriting them with release binaries.
- Treating a successful download as a successful application update.

## 14. CtrlSpeak-specific follow-up questions

The Handy review suggests several decisions for future planning:

1. Should the Ubuntu server report its CtrlSpeak protocol/application version in
   its health response so the Windows client can identify incompatible pairs?
2. Should client and server releases share a version but use separate artifacts
   and manifest roles?
3. Should update checks remain opt-in, become opt-out, or run only when the user
   opens the management window?
4. Should CtrlSpeak expose a portable-data mode, or standardize exclusively on
   per-user application-data storage?
5. Is Copy Last Transcript sufficient as the first history feature, avoiding
   persistent transcript storage altogether?
6. Which Linux package should eventually become the supported self-update unit:
   a standalone executable, AppImage, deb/rpm package, or a service deployment?
7. Should remote-server updates remain an administrator/SSH operation rather
   than being initiated by a Windows client?

These are intentionally deferred from the v0.5 critical path unless the update
architecture requires an answer.

## 15. Source map for future re-review

The following Handy files were the most relevant at the reviewed commit:

- `src/components/update-checker/UpdateChecker.tsx` — update UI states, manual
  versus background checks, progress, installation, and relaunch.
- `src/components/update-checker/portableInstaller.ts` — exact manifest target
  selection and immutable installer URLs for portable Windows use.
- `src/components/update-checker/portableInstaller.test.ts` — target-selection
  assertions and malformed-manifest fallbacks.
- `src/components/whats-new/WhatsNewGate.tsx` — one-time release-note gate.
- `src/components/whats-new/releaseNotes.ts` — versioned bundled notes.
- `src/stores/modelStore.ts` — reconciliation and lifecycle state for downloads.
- `src/stores/settingsStore.ts` — centralized settings updates.
- `src-tauri/src/managers/model/download.rs` — resume, timeout, cancellation,
  size bounding, progress throttling, and verification.
- `src-tauri/src/portable.rs` — early runtime-mode detection and portable-aware
  path resolution.
- `src-tauri/src/settings.rs` — typed settings/defaults and persistence.
- `src-tauri/src/tray.rs` — state-dependent tray menus, version, and update item.
- `src-tauri/tauri.conf.json` — updater key, endpoint, packaging, and platform
  metadata.

When revisiting Handy, pin a new commit and compare it with the snapshot above.
The upstream implementation will evolve, and later behavior should not be
assumed to match this review.
