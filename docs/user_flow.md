# CtrlSpeak v0.6.2 user flow

## 1. Launch and platform readiness

1. CtrlSpeak parses maintenance/backend/update arguments, acquires the per-user
   instance lock, salvages and migrates `settings.json` field by field, validates
   the backend URL, and pins the effective backend for the process.
2. The splash and hidden Tk root are created on the main thread.
3. When the tray runtime starts, the Linux path validates the desktop session
   before importing/starting `pynput`. X11 continues; Wayland or a missing
   `DISPLAY` leaves the client listener stopped and shows an actionable Xorg
   message.
4. Tk remains the main UI loop while `pystray` runs its Linux backend in a
   dedicated event thread. Tray failure reports the limitation and opens the
   management window so the user is not left with an unreachable background
   process.

All persistent data is below `$XDG_CONFIG_HOME/CtrlSpeak` or
`~/.config/CtrlSpeak`. Packaged assets are read-only.

## 2. Backend startup branches

### Embedded / local

1. CtrlSpeak prompts once for the legacy **Client + Server** or **Client Only**
   role if no mode is stored.
2. The selected Whisper model is stored below the XDG model directory. CPU and
   `small` remain the safe defaults.
3. Client + Server loads the local model and starts the legacy HTTP/discovery
   threads. Client Only records locally and sends to the discovered or manually
   configured trusted-LAN CtrlSpeak server.
4. Local CUDA is selected only if the Linux NVIDIA driver and CTranslate2
   runtime probe succeeds. Linux driver/runtime installation is external.

### Remote API

1. Legacy role selection, discovery, local server startup, model download, and
   model warm-up are skipped.
2. The configured HTTP(S) base URL and optional in-memory bearer token are used
   directly. No LAN address is hardcoded.
3. **Check gateway** reads authenticated capabilities, rejects a worker-only
   endpoint, and populates the gateway's published provider strategies.
4. An optional OpenAI key may be stored for the current Windows user in Windows
   Credential Manager and is loaded into the desktop process at startup. It is
   sent on each relevant request but never written to `settings.json`, retained
   by the gateway, or sent to the Ubuntu worker. Unsupported platforms remain
   session-only.

## 3. Recording

1. A non-suppressing global `pynput` listener observes the right Ctrl press.
2. CtrlSpeak creates a unique WAV below the XDG `temp/` directory and opens the
   selected/default PortAudio input device.
3. The waveform overlay updates while the key is held.
4. Releasing right Ctrl stops recording, changes the overlay to Processing, and
   starts the processing sound. The packaged chime is uniformly attenuated when
   necessary to remain at or below the -6 dBFS peak ceiling.
5. Audio/open-device errors are logged and reported; they do not leave a
   recording beside the executable.

## 4. Transcription and injection

### Embedded result

The configured `faster-whisper` model transcribes locally (or the legacy
Client-Only role calls its trusted-LAN server). The exact local correction
library applies only a previously approved byte-for-byte raw match. A new local
result receives a local ID and feedback target.

With no output languages selected, Whisper remains automatic and unrestricted.
One selected language is forced. With two to five selected languages, CtrlSpeak
accepts detection only inside that ordered allowlist and otherwise forces its
first entry. A model-reported language outside the policy is refused.

### API result

CtrlSpeak uploads multipart WAV audio to `<base-url>/v1/transcribe`. The result
retains `id`, `raw_text`, corrected `text`, and all response metadata. HTTP,
authentication, network, or schema errors are shown and never trigger embedded
fallback.

The gateway may use its selected published cascade. `provider_used`, `attempts`,
and `degraded` make that routing visible. The default `ubuntu-gpu-preferred`
route is Ubuntu GPU, then OpenAI with the caller's key, then gateway tiny.
`openai-preferred` reverses the first two; Ubuntu-only, OpenAI-only, and
gateway-tiny-only routes are also available. Cascades continue after quota/key
failures and record them; single-provider routes return them. An offline Ubuntu
worker is probed within 500 ms and then bypassed through a 30-second circuit
instead of delaying every request.

When configured, the ordered language policy is sent as the multipart
`allowed_languages` field. The maintained server validates and enforces
it, and the client independently refuses an out-of-policy response.

### Linux insertion

1. On X11, CtrlSpeak prefers `xclip`: it snapshots the prior text clipboard,
   stages the transcript, sends Ctrl+V through `pyautogui`, waits briefly for
   the target, and restores the prior text.
2. If `xclip` is missing, an included withdrawn Tk root performs the same
   transaction for Unicode and multiline text while processing the X11
   selection events needed to serve the paste. The root is cleaned up after
   restoration.
3. If neither provider is usable, an actionable `DISPLAY`/tkinter error is
   reported. A non-text clipboard is not overwritten; only safe single-line
   ASCII direct typing is then attempted. Native Wayland stops at the earlier
   capability check.
4. Before insertion, CtrlSpeak retains the successful text as the one in-memory
   **Copy last transcript** recovery value. Only after successful injection does
   it make the result eligible for edit feedback.

The processing sound/overlay stops and the temporary WAV is cleaned in every
result path.

### Tray recovery

The tray's **Copy last transcript** item is disabled until transcription first
succeeds. It copies the retained text even if active-field insertion failed and
reports the clipboard outcome. CtrlSpeak retains exactly one result in memory;
it does not persist transcript history, and the value disappears on exit.

### Tray correction submission

The tray's **Submit correction…** action opens one small form. Enter the phrase
CtrlSpeak currently produces and the replacement it should return. Submission
uses the runtime-pinned gateway URL and bearer identity; the token is never
shown in the form or logged. The default user-scoped rule affects only that
identity. An administrator may opt into a global rule for all gateway users.
The dialog validates empty and unchanged pairs, performs the network request in
the background, and confirms whether the rule became active.

Embedded/local mode explains that a remote gateway must be selected instead of
pretending to save a server correction locally.

## 6. Best-effort correction feedback

1. The latest eligible injection remains pending for up to ten minutes.
2. Shift/Alt/Ctrl-modified Enter, releases, and unrelated keys do not consume it.
3. On bare Enter press, before returning from the observer callback, CtrlSpeak
   uses X11 Ctrl+A/C to snapshot the field through preferred `xclip` or the
   tkinter fallback when a safely restorable text clipboard is available.
4. The prior text clipboard is restored. CtrlSpeak never sends Enter itself;
   the target receives the user's original Enter exactly once.
5. Unchanged, empty, expired, or unavailable snapshots are discarded.
6. A changed local result is stored as an exact override. A changed API result
   is sent to the original
   `<base-url>/v1/transcriptions/{id}/feedback`, even if saved settings changed
   after injection.

The field selection/copy technique cannot guarantee support for every toolkit,
terminal, remote surface, password field, or multiline editor. Disable edit
feedback in the management window when the workflow is unsuitable.

## 7. Management window

The tray's **Manage CtrlSpeak** action raises one Tk management window. It
contains:

- current version, stable update channel, last-check time, signed update status,
  download progress, release link, and update controls;
- embedded/API selection, output-language multi-select, complete base URL,
  masked client token, gateway capability check/strategy, session-only masked
  OpenAI key, feedback method, and redacted active/saved status;
- legacy embedded role/network controls;
- microphone selection;
- model selection/download;
- CPU/GPU preference and Linux **Recheck system CUDA** status;
- client start/stop and clean application shutdown.

Backend changes explicitly require restart. On Linux, the destructive Windows
**Delete CtrlSpeak** flow only reports that automatic uninstall is unsupported;
the user/operator removes their manually installed files.

## 7. Signed update flow

1. **Check for updates** queries GitHub's latest stable CtrlSpeak Release on a
   worker thread. The tray and control center share one generation-aware
   coordinator, so a late older request cannot overwrite a newer result.
2. CtrlSpeak downloads and verifies the canonical manifest and Ed25519
   signature, then selects exactly one standard artifact for the current OS and
   x86-64 architecture.
3. After the user confirms, the artifact streams into an AppData/XDG transaction
   directory. Compatible partials resume only with a correct HTTP Range response.
4. Declared size and SHA-256 must match before the candidate becomes installable.
5. A second confirmation starts a copied external helper. CtrlSpeak stops its
   hotkey, discovery/server, tray, UI, and instance lock, then exits normally.
6. The helper preserves a verified previous executable, atomically swaps the
   stable installed filename, and launches the new version.
7. The new process confirms its transaction, version, path, settings migration,
   backend activation, and readiness. If it cannot do so, the helper restores
   and relaunches the previous executable.
8. A successful update may show bounded plain-text What's New information once.
   A rollback reports restoration instead of claiming success.

Source checkouts are discovery-only. They never run Git commands or overwrite
working files from the GUI.

## 8. Shutdown

Quit stops the hotkey/recording path, hides overlays, cleans temporary audio,
stops embedded discovery/server threads, stops the tray, exits Tk, and releases
the XDG instance lock. No service, firewall, desktop launcher, or remote API
deployment is modified.
