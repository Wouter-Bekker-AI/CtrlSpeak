# CtrlSpeak v0.4 Linux user flow

## 1. Launch and platform readiness

1. CtrlSpeak parses maintenance/backend arguments, acquires the per-user XDG
   instance lock, loads `settings.json`, validates the backend URL, and pins the
   effective backend for the process.
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
   directly. No LAN address is hardcoded and no backend fallback occurs.

## 3. Recording

1. A non-suppressing global `pynput` listener observes the right Ctrl press.
2. CtrlSpeak creates a unique WAV below the XDG `temp/` directory and opens the
   selected/default PortAudio input device.
3. The waveform overlay updates while the key is held.
4. Releasing right Ctrl stops recording, changes the overlay to Processing, and
   starts the processing sound.
5. Audio/open-device errors are logged and reported; they do not leave a
   recording beside the executable.

## 4. Transcription and injection

### Embedded result

The configured `faster-whisper` model transcribes locally (or the legacy
Client-Only role calls its trusted-LAN server). The exact local correction
library applies only a previously approved byte-for-byte raw match. A new local
result receives a local ID and feedback target.

### API result

CtrlSpeak uploads multipart WAV audio to `<base-url>/v1/transcribe`. The result
retains `id`, `raw_text`, corrected `text`, and all response metadata. HTTP,
authentication, network, or schema errors are shown and never trigger embedded
fallback.

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
4. Only after successful injection does CtrlSpeak make the result eligible for
   edit feedback.

The processing sound/overlay stops and the temporary WAV is cleaned in every
result path.

## 5. Best-effort correction feedback

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

## 6. Management window

The tray's **Manage CtrlSpeak** action raises one Tk management window. It
contains:

- embedded/API selection, complete base URL, masked token, feedback method, and
  redacted active/saved status;
- legacy embedded role/network controls;
- microphone selection;
- model selection/download;
- CPU/GPU preference and Linux **Recheck system CUDA** status;
- client start/stop and clean application shutdown.

Backend changes explicitly require restart. On Linux, the destructive Windows
**Delete CtrlSpeak** flow only reports that automatic uninstall is unsupported;
the user/operator removes their manually installed files.

## 7. Shutdown

Quit stops the hotkey/recording path, hides overlays, cleans temporary audio,
stops embedded discovery/server threads, stops the tray, exits Tk, and releases
the XDG instance lock. No service, firewall, desktop launcher, or remote API
deployment is modified.
