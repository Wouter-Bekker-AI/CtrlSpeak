# Building CtrlSpeak v0.4 for Linux

This is the maintained Ubuntu/Linux packaging path. It produces
`dist/CtrlSpeak_v0.4`; it does not install the result, create a desktop
launcher, start a service, deploy an API, or alter firewall policy.

## Prerequisites

Build on a 64-bit Linux host. PyInstaller builds for the host operating system;
it does not cross-compile the Linux executable from Windows.
Build on the oldest Ubuntu/glibc release you intend to support because
PyInstaller does not bundle Linux `libc`; a bundle made on a newer distribution
may not start on an older one.

Typical Ubuntu packages are:

```bash
sudo apt install python3-venv python3-dev python3-tk portaudio19-dev \
  libportaudio2 xclip libx11-6 libxtst6 libxinerama1 libxrandr2 libxi6
```

Create a clean environment and install Python dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`xclip` is preferred for clipboard-preserving paste and automatic active-field
edit capture, but is not a hard runtime prerequisite. When it is absent, the
included tkinter runtime provides a withdrawn Tk/X11 clipboard fallback with
Unicode/multiline staging, selection-event servicing, and prior-text
restoration. If neither provider can access the active X11 display, CtrlSpeak
reports actionable `DISPLAY`/tkinter guidance; only plain single-line ASCII can
use direct typing. PortAudio and a working desktop input source are required
for recording. Ubuntu GNOME may require AppIndicator/legacy tray support before
the tray icon is visible.

The global hotkey and injection path requires an X11 session. At the Ubuntu
sign-in screen choose **Ubuntu on Xorg**. Native Wayland is intentionally
reported as unsupported.

## Build

From the project root:

```bash
python -m utils.build_exe
```

The helper selects `packaging/CtrlSpeak_v0.4.spec`. The equivalent direct
command is:

```bash
pyinstaller --noconfirm --clean packaging/CtrlSpeak_v0.4.spec
```

The one-file spec uses `console=False`, includes the native PNG icon and shipped
assets, collects the Whisper/CTranslate2/HTTP runtime data, includes the X11
`pynput` and Linux `pystray` backends, and explicitly excludes the lazy Windows
input adapter. Whisper weights and Linux system CUDA/cuDNN libraries are not
bundled.

Expected artifact:

```bash
test -x dist/CtrlSpeak_v0.4
file dist/CtrlSpeak_v0.4
```

Do not claim release readiness from a successful PyInstaller command alone.
Run the source/headless checks and the physical desktop checklist in
`docs/TESTING.md` against the exact artifact.

## Uninstalled desktop and AppStream metadata

`packaging/linux/ctrlspeak.desktop` is a template. Its executable and icon
placeholders must be replaced with absolute paths. The following is an example
of a user-local installation; do not run it as part of the build:

```bash
install -Dm755 dist/CtrlSpeak_v0.4 \
  "$HOME/.local/opt/ctrlspeak/CtrlSpeak_v0.4"
install -Dm644 assets/icon.png \
  "$HOME/.local/share/icons/hicolor/128x128/apps/ctrlspeak.png"
sed \
  -e "s|@CTRLSPEAK_EXECUTABLE@|$HOME/.local/opt/ctrlspeak/CtrlSpeak_v0.4|g" \
  -e "s|@CTRLSPEAK_ICON@|$HOME/.local/share/icons/hicolor/128x128/apps/ctrlspeak.png|g" \
  packaging/linux/ctrlspeak.desktop \
  > /tmp/ctrlspeak.desktop
desktop-file-validate /tmp/ctrlspeak.desktop
install -Dm644 /tmp/ctrlspeak.desktop \
  "$HOME/.local/share/applications/ctrlspeak.desktop"
```

The AppStream source is
`packaging/linux/io.trueai.ctrlspeak.metainfo.xml`. Validate it when
`appstreamcli` is available:

```bash
appstreamcli validate --no-net packaging/linux/io.trueai.ctrlspeak.metainfo.xml
```

No launcher or metadata has been installed merely because these templates are
present in the repository.

## Linux CUDA boundary

CPU is the default. CtrlSpeak detects `libcuda.so.1` and asks CTranslate2
whether a CUDA device is usable. The v0.4 Linux app does not install NVIDIA
drivers, CUDA, cuDNN, modify loader configuration, or change system services.
Provide a driver/runtime combination compatible with the installed
CTranslate2 build, then use **Recheck system CUDA** or:

```bash
python main.py --download-cuda-only
```

On Linux that command is a readiness check only. It exits non-zero with an
actionable message if the system runtime is not usable.

## Known desktop limitations

- Native Wayland global hotkeys and injection are unsupported.
- X11 Ctrl+A/C feedback cannot read every toolkit/control and treats the
  selected field as the complete final transcript.
- Non-text clipboard data is preserved by avoiding clipboard replacement;
  Unicode/multiline injection then requires the operation to stop.
- Tray visibility depends on the desktop shell's indicator support.
- A one-file PyInstaller program extracts to a temporary runtime directory on
  launch and cannot start when that extraction location is mounted `noexec`;
  persistent state still goes only to the XDG CtrlSpeak directory.

## Legacy Windows build preservation

The existing Windows specifications remain unchanged. On Windows,
`python -m utils.build_exe` still selects `packaging/CtrlSpeak_v0.3.spec` and
produces `CtrlSpeak_v0.3.exe` with `console=False`; the executable does not have a console window.
The Watcher option also remains Windows-only. The Linux v0.4
work does not build, install, or modify a Windows workspace.
