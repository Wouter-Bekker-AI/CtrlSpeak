# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Optional
from queue import Empty

import tkinter as tk
from tkinter import ttk, messagebox
import pystray
import numpy as np


# ---- System-layer pieces (lifecycle, discovery, status) ----
from utils.system import (
    APP_VERSION,
    CLIENT_ONLY_BUILD,
    settings, settings_lock, load_settings,
    detect_client_only_build, save_settings, notify,
    start_client_listener, stop_client_listener,
    manual_discovery_refresh,
    schedule_management_refresh, enqueue_management_task,
    shutdown_server, initiate_self_uninstall,
    resource_path,
)
# IMPORTANT: import the module so we always see the *current* values
from utils import system as sysmod

# ---- Model/CUDA helpers kept in utils.models to avoid GUI bloat ----
from utils.models import (
    cuda_runtime_ready,
    install_cuda_runtime_with_progress,   # opens progress window and installs nvidia wheels
    get_device_preference, set_device_preference,
    get_current_model_name, set_current_model_name,
    model_store_path_for, model_files_present,
    download_model_with_gui,              # opens progress window and downloads model
    format_bytes,                         # optional pretty bytes
)

# Shared UI thread root + instance ref (imported by utils.system.schedule_management_refresh)
tk_root: Optional[tk.Tk] = None
management_window: Optional["ManagementWindow"] = None

# -------- Voice Waveform Overlay --------
_waveform_win: Optional[tk.Toplevel] = None
_waveform_canvas: Optional[tk.Canvas] = None
_waveform_job: Optional[str] = None
_waveform_provider: Optional[Callable[[], "np.ndarray"]] = None

# NEW: simple state for live vs processing
_waveform_mode: str = "live"         # "live" | "processing"
_waveform_msg: str = "Processing…"
_pulse_phase: float = 0.0            # animation phase
_waveform_closing: bool = False


def show_waveform_overlay(provider: Callable[[], "np.ndarray"]) -> None:
    global _waveform_win, _waveform_canvas, _waveform_job, _waveform_provider, _waveform_mode, _waveform_closing
    _waveform_provider = provider
    _waveform_mode = "live"
    try:
        if tk_root is None or not tk_root.winfo_exists():
            return
        if _waveform_win and _waveform_win.winfo_exists():
            return
        _waveform_win = tk.Toplevel(tk_root)
        _waveform_win.overrideredirect(True)
        _waveform_win.attributes("-topmost", True)
        try:
            _waveform_win.attributes("-alpha", 0.92)  # translucent
        except Exception:
            pass

        # Position: top half center
        try:
            sw, sh = tk_root.winfo_screenwidth(), tk_root.winfo_screenheight()
        except Exception:
            sw, sh = 1200, 800
        target_w = int(sw * 0.4)
        target_h = int(sh * 0.25)
        x = (sw - target_w) // 2
        y = int(sh * 0.05)
        _waveform_win.geometry(f"{target_w}x{target_h}+{x}+{y}")

        _waveform_canvas = tk.Canvas(_waveform_win, bg="#141414", highlightthickness=0)
        _waveform_canvas.pack(fill=tk.BOTH, expand=True)
        _waveform_closing = False

        def _tick():
            """Redraw the overlay every ~33 ms.
            - LIVE mode: polyline waveform from recent audio samples.
            - PROCESSING mode: pulsing circular ring with 'Processing…' label.
            """
            global _pulse_phase
            if _waveform_win is None or not _waveform_win.winfo_exists():
                return
            if _waveform_canvas is None:
                return

            try:
                w = max(1, _waveform_canvas.winfo_width())
                h = max(1, _waveform_canvas.winfo_height())

                _waveform_canvas.delete("all")
                # background panel
                _waveform_canvas.create_rectangle(12, 12, w - 12, h - 12, fill="#202020", outline="#333333")

                if _waveform_mode == "live":
                    if _waveform_provider is not None:
                        data = _waveform_provider()
                        if data is not None and getattr(data, "size", 0) > 0:
                            # DYNAMIC SCALE WITH UPPER CAP
                            # Aim to keep the current frame’s peak around ~0.9, but never amplify above 8x.
                            TARGET_PEAK = 0.90
                            MAX_GAIN = 8.0

                            m = float(np.max(np.abs(data))) if getattr(data, "size", 0) else 0.0
                            if m > 1e-6:
                                dyn_gain = min(MAX_GAIN, TARGET_PEAK / m)
                            else:
                                dyn_gain = 1.0  # silence; no crazy boost

                            arr = np.clip(data * dyn_gain, -1.0, 1.0)

                            # downsample to ~canvas width
                            count = max(2, w - 40)
                            idxs = np.linspace(0, arr.size - 1, num=count).astype(int)
                            ys = arr[idxs]
                            mid = h / 2.0
                            amp = (h - 48) / 2.0

                            last_x, last_y = None, None
                            for i, v in enumerate(ys):
                                X = 20 + i
                                Y = int(mid - v * amp)
                                if last_x is not None:
                                    _waveform_canvas.create_line(last_x, last_y, X, Y, fill="#6ee7ff", width=2)
                                last_x, last_y = X, Y

                else:
                    # PROCESSING — audio-driven radius + circular wiggle
                    _pulse_phase = (_pulse_phase + 0.06) % (2 * np.pi)  # subtle motion only
                    # 1) Read the current amplitude + recent waveform of loading.wav

                    try:
                        level = float(sysmod.get_processing_level())
                        proc_wave = sysmod.get_processing_waveform(720)  # resolution around the ring
                    except Exception:
                        level = 0.0
                        proc_wave = np.zeros(720, dtype=np.float32)

                    # Normalize/soft-clip the ring waveform
                    if proc_wave.size:
                        m = float(np.max(np.abs(proc_wave))) or 1.0
                        ring_wave = np.clip(proc_wave / m, -1.0, 1.0)
                    else:
                        ring_wave = np.zeros(720, dtype=np.float32)

                    # 2) Map amplitude to base radius scaling (more “one-to-one” feel)
                    #    Increase LEVEL_GAIN to make size swings stronger (try 0.6–1.0)
                    LEVEL_GAIN = 0.75
                    pulse = 1.0 + LEVEL_GAIN * level

                    # 3) Wiggle strength around the ring (how spiky the line looks)
                    #    Try 0.20–0.40 for pronounced wiggle
                    WIGGLE_GAIN = 0.30
                    w = max(1, _waveform_canvas.winfo_width())
                    h = max(1, _waveform_canvas.winfo_height())
                    cx, cy = w // 2, h // 2
                    base_r = int(min(w, h) * 0.20)
                    ring_thickness = 8

                    # Base radius from amplitude
                    R = int(base_r * pulse * 1.06)
                    # 4) Build a closed polyline around the circle with radius modulation
                    #    by the audio waveform (and a tiny phase spin so it feels alive)
                    N = ring_wave.size

                    points = []

                    for i in range(N):
                        a = (2 * np.pi * i) / N + _pulse_phase * 0.5
                        # radius wiggle from waveform
                        r = R + int(WIGGLE_GAIN * base_r * ring_wave[i])
                        x = cx + int(np.cos(a) * r)
                        y = cy + int(np.sin(a) * r)
                        points.append((x, y))

                    # Draw the wiggly ring
                    for i in range(1, len(points)):
                        x0, y0 = points[i - 1]
                        x1, y1 = points[i]
                        _waveform_canvas.create_line(x0, y0, x1, y1, fill="#6ee7ff", width=2)

                    # close the loop
                    if len(points) > 2:
                        _waveform_canvas.create_line(points[-1][0], points[-1][1], points[0][0], points[0][1],
                                                     fill="#6ee7ff", width=2)

                    # (Optional) inner glow ring following the same R for body

                    r_inner = max(4, R - ring_thickness)
                    _waveform_canvas.create_oval(cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
                                                 outline="#6ee7ff", width=1)
                    # label
                    _waveform_canvas.create_text(cx, cy, text=_waveform_msg, fill="#d9fbff",
                                                 font=("Segoe UI", 16, "bold"))

                if not _waveform_closing and _waveform_win and _waveform_win.winfo_exists():
                    _waveform_job = _waveform_canvas.after(33, _tick)

            except Exception:
                if not _waveform_closing and _waveform_win and _waveform_win.winfo_exists():
                    _waveform_job = _waveform_canvas.after(33, _tick)

        _tick()
    except Exception:
        pass

def set_waveform_processing(message: str = "Processing…") -> None:
    global _waveform_mode, _waveform_msg, _pulse_phase
    _waveform_mode = "processing"
    _waveform_msg = message
    _pulse_phase = 0.0   # ← new

def hide_waveform_overlay() -> None:
    global _waveform_win, _waveform_canvas, _waveform_job, _waveform_provider
    global _waveform_mode, _waveform_msg, _waveform_closing
    try:
        _waveform_closing = True
        if _waveform_canvas and _waveform_job:
            try:
                _waveform_canvas.after_cancel(_waveform_job)
            except Exception:
                pass
        if _waveform_win and _waveform_win.winfo_exists():
            try:
                _waveform_win.withdraw()
            except Exception:
                pass
            # destroy shortly after to let any in-flight callbacks finish
            _waveform_win.after(10, _waveform_win.destroy)
    finally:
        _waveform_win = None
        _waveform_canvas = None
        _waveform_job = None
        _waveform_provider = None
        _waveform_mode = "live"
        _waveform_msg = "Processing…"

# ---------------- Splash (1s) ----------------
def show_splash_screen(duration_ms: int) -> None:
    try:
        root = tk.Tk()
    except tk.TclError:
        return
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(background="#1a1a1a")
    width, height = 260, 240
    try:
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
    except Exception:
        screen_w, screen_h = 800, 600
    pos_x = int((screen_w - width) / 2); pos_y = int((screen_h - height) / 2)
    root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")

    try:
        from PIL import Image, ImageTk
        icon_path = resource_path("icon.ico")
        image = Image.open(icon_path)
        image.thumbnail((160, 160), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(root, image=photo, background="#1a1a1a")
        label.image = photo
        label.pack(pady=(32, 12))
    except Exception:
        tk.Label(root, text="CtrlSpeak", font=("Segoe UI", 18, "bold"),
                 fg="#ffffff", bg="#1a1a1a").pack(pady=(40, 20))
    tk.Label(root, text="Loading...", font=("Segoe UI", 11),
             fg="#dddddd", bg="#1a1a1a").pack()
    root.after(duration_ms, root.destroy)
    root.mainloop()

# ---------------- First-run mode ----------------
def prompt_initial_mode() -> Optional[str]:
    from utils.system import AUTO_MODE_PROFILE
    if AUTO_MODE_PROFILE:
        return AUTO_MODE_PROFILE
    result = {"mode": None}

    def choose(mode: str) -> None:
        result["mode"] = mode
        root.destroy()

    root = tk.Tk()
    root.title("Welcome to CtrlSpeak")
    root.resizable(True, True)
    root.attributes("-topmost", True)

    tk.Label(root, text="Welcome to CtrlSpeak",
             font=("Segoe UI", 14, "bold")).pack(pady=(18, 8))
    message = (
        "It looks like this is the first time CtrlSpeak is running on this computer.\n\n"
        "Choose how you would like to use it:"
    )
    tk.Message(root, text=message, width=420, font=("Segoe UI", 10)).pack(pady=(0, 12))

    button_frame = tk.Frame(root)
    button_frame.pack(pady=(0, 12), fill=tk.X, padx=16)

    def make_button(text: str, desc: str, mode_value: str) -> None:
        frame = tk.Frame(button_frame, relief=tk.RIDGE, borderwidth=1)
        frame.pack(padx=8, pady=6, fill=tk.X)
        tk.Label(frame, text=text, font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 0))
        tk.Message(frame, text=desc, width=360).pack(anchor=tk.W, padx=8)
        tk.Button(frame, text="Select", command=lambda value=mode_value: choose(value)).pack(padx=8, pady=(0, 8), anchor=tk.E)

    make_button("Client + Server",
                "Use this computer for local transcription and share it with other CtrlSpeak clients on the network.",
                "client_server")
    make_button("Client Only",
                "Send recordings to another CtrlSpeak server on your local network for transcription.",
                "client")

    def cancel() -> None: root.destroy()
    tk.Button(root, text="Quit", command=cancel).pack(pady=(0, 12))
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.update_idletasks()
    req_w = root.winfo_reqwidth()
    req_h = root.winfo_reqheight()
    root.geometry(f"{req_w}x{req_h}")
    root.minsize(req_w, req_h)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return result["mode"]


def ensure_mode_selected() -> None:
    # 1) Load settings from disk first
    load_settings()

    # 2) If mode already set, don't prompt again
    with settings_lock:
        current_mode = settings.get("mode")

    if current_mode in {"client", "client_server"}:
        return  # nothing to do

    # 3) Client-only builds force 'client' once and persist
    if detect_client_only_build():
        with settings_lock:
            settings["mode"] = "client"
        save_settings()
        return

    # 4) First-run prompt only if still not set
    choice = prompt_initial_mode()
    if not choice:
        notify("CtrlSpeak cannot continue without selecting a mode.")
        sys.exit(0)
    with settings_lock:
        settings["mode"] = choice
    save_settings()

# ---------------- UI loop pump (for async updates) ----------------
def ensure_management_ui_thread() -> None:
    global tk_root
    if tk_root and tk_root.winfo_exists():
        return

    def _loop():
        global tk_root
        tk_root = tk.Tk()
        tk_root.withdraw()

        def process_queue() -> None:
            # pull queue from system module (so it’s always the live one)
            while True:
                try:
                    func, args, kwargs = sysmod.management_ui_queue.get_nowait()
                except Empty:
                    break
                try:
                    func(*args, **kwargs)
                except Exception:
                    import traceback
                    traceback.print_exc()
            if tk_root is not None:
                tk_root.after(120, process_queue)

        tk_root.after(80, process_queue)
        tk_root.mainloop()

    t = threading.Thread(target=_loop, name="CtrlSpeakManagementUI", daemon=True)
    t.start()

# ---------------- Status helper ----------------
def describe_server_status() -> str:
    with settings_lock:
        mode = settings.get("mode")
        port = int(settings.get("server_port", 65432))

    if mode == "client_server" and sysmod.server_thread and sysmod.server_thread.is_alive():
        host = sysmod.last_connected_server.host if sysmod.last_connected_server else sysmod.get_advertised_host_ip()
        return f"Serving: {host}:{port}"

    if sysmod.last_connected_server:
        host = sysmod.last_connected_server.host
        prt = sysmod.last_connected_server.port
        label = "local CPU" if host == "local-cpu" else ("local" if host == "local" else f"{host}:{prt}")
        return f"Connected: {label}"

    server = sysmod.get_best_server()
    if server:
        return f"Discovered: {server.host}:{server.port}"

    return "Not connected"

# ---------------- Management window (NEW UI) ----------------
def open_management_dialog(icon, item):
    ensure_management_ui_thread()
    enqueue_management_task(_show_management_window, icon)

def _show_management_window(icon: pystray.Icon) -> None:
    global management_window
    if management_window and management_window.is_open():
        management_window.bring_to_front()
        management_window.refresh_status()
        return
    management_window = ManagementWindow(icon)

class ManagementWindow:
    def __init__(self, icon: pystray.Icon):
        self._icon = icon
        self.window = tk.Toplevel(tk_root)
        self.window.title(f"CtrlSpeak Control v{APP_VERSION}")
        self.window.geometry("560x520")
        self.window.minsize(520, 480)
        self.window.resizable(True, True)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _e: self.close())
        try:
            self.window.iconbitmap(resource_path("icon.ico"))
        except Exception:
            pass

        wrap = ttk.Frame(self.window, padding=(16, 14)); wrap.pack(fill=tk.BOTH, expand=True)
        ttk.Label(wrap, text="CtrlSpeak Control", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        build_label = "Client Only" if CLIENT_ONLY_BUILD else "Client + Server"
        ttk.Label(wrap, text=f"Build {APP_VERSION} - {build_label}", font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 10))

        # Status
        self.status_var = tk.StringVar()
        self.server_status_var = tk.StringVar()
        ttk.Label(wrap, textvariable=self.status_var, justify=tk.LEFT, anchor="w").pack(fill=tk.X, pady=(0, 6))
        ttk.Label(wrap, textvariable=self.server_status_var, font=("Segoe UI", 10, "italic"),
                  justify=tk.LEFT, anchor="w").pack(fill=tk.X, pady=(0, 10))

        # Device (CPU/GPU)
        device_frame = ttk.LabelFrame(wrap, text="Device")
        device_frame.pack(fill=tk.X, pady=(0, 10))
        self.device_var = tk.StringVar(value=get_device_preference())
        row = ttk.Frame(device_frame); row.pack(fill=tk.X, padx=6, pady=6)
        ttk.Radiobutton(row, text="CPU", variable=self.device_var, value="cpu").pack(side=tk.LEFT, padx=(0,12))
        ttk.Radiobutton(row, text="GPU (CUDA)", variable=self.device_var, value="cuda").pack(side=tk.LEFT)
        self.cuda_status = tk.StringVar()
        ttk.Label(device_frame, textvariable=self.cuda_status).pack(anchor="w", padx=6, pady=(2,8))
        btns_dev = ttk.Frame(device_frame); btns_dev.pack(fill=tk.X, padx=6, pady=(0,10))
        self.apply_device_btn = ttk.Button(btns_dev, text="Apply Device", command=self._apply_device)
        self.apply_device_btn.pack(side=tk.LEFT)
        self.install_cuda_btn = ttk.Button(btns_dev, text="Install/Repair CUDA…", command=self._install_cuda)
        self.install_cuda_btn.pack(side=tk.LEFT, padx=(8,0))

        # Model selector
        model_frame = ttk.LabelFrame(wrap, text="Model")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        self.model_var = tk.StringVar(value=get_current_model_name())
        rowm = ttk.Frame(model_frame); rowm.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(rowm, text="Whisper model:").pack(side=tk.LEFT)
        ttk.Combobox(rowm, textvariable=self.model_var, values=["small", "large-v3"],
                     state="readonly", width=18).pack(side=tk.LEFT, padx=(8,0))
        btns_m = ttk.Frame(model_frame); btns_m.pack(fill=tk.X, padx=6, pady=(6,10))
        self.apply_model_btn = ttk.Button(btns_m, text="Set Model", command=self._apply_model)
        self.apply_model_btn.pack(side=tk.LEFT)
        self.download_model_btn = ttk.Button(btns_m, text="Download/Update…", command=self._download_model)
        self.download_model_btn.pack(side=tk.LEFT, padx=(8,0))
        self.model_status = tk.StringVar()
        ttk.Label(model_frame, textvariable=self.model_status).pack(anchor="w", padx=6, pady=(4,0))

        # Client/Server controls
        ctrl = ttk.LabelFrame(wrap, text="Client & Server")
        ctrl.pack(fill=tk.X, pady=(0,10))
        self.stop_button = ttk.Button(ctrl, text="Stop Client", command=self.stop_client); self.stop_button.pack(fill=tk.X, pady=2, padx=6)
        self.start_button = ttk.Button(ctrl, text="Start Client", command=self.start_client); self.start_button.pack(fill=tk.X, pady=2, padx=6)
        self.refresh_button = ttk.Button(ctrl, text="Refresh Servers", command=self.refresh_servers); self.refresh_button.pack(fill=tk.X, pady=2, padx=6)
        exit_label = "Exit CtrlSpeak" if CLIENT_ONLY_BUILD else "Stop Everything"
        self.stop_all_button = ttk.Button(ctrl, text=exit_label, command=self.stop_everything)
        self.stop_all_button.pack(fill=tk.X, pady=(8, 6), padx=6)

        ttk.Separator(wrap).pack(fill=tk.X, pady=(8, 10))
        ttk.Button(wrap, text="Close", command=self.close).pack(fill=tk.X)
        delete_btn = tk.Button(wrap, text="Delete CtrlSpeak", command=self.delete_ctrlspeak,
                               bg="#d32f2f", fg="white", activebackground="#b71c1c", activeforeground="white")
        delete_btn.pack(fill=tk.X, pady=(6,0))

        self.window.after(120, self.refresh_status)

        # --- Auto-size window to fit content once everything is laid out ---
        self.window.update_idletasks()  # compute requested size
        req_w = self.window.winfo_reqwidth()
        req_h = self.window.winfo_reqheight()
        # Don’t let geometry be smaller than required; also set minsize to prevent clipping
        self.window.geometry(f"{req_w}x{req_h}")
        self.window.minsize(req_w, req_h)

        self.bring_to_front()

    # --- window helpers ---
    def is_open(self) -> bool:
        return bool(self.window and self.window.winfo_exists())

    def bring_to_front(self) -> None:
        if not self.is_open(): return
        self.window.deiconify(); self.window.lift(); self.window.focus_force()
        self.window.attributes("-topmost", True)
        self.window.after(150, lambda: self.window.attributes("-topmost", False))

    # --- state refresh ---
    def refresh_status(self) -> None:
        with settings_lock:
            mode = settings.get("mode") or "unknown"
        device_pref = get_device_preference()
        cuda_ok = cuda_runtime_ready()
        model_name = get_current_model_name()
        present = model_files_present(model_store_path_for(model_name))
        status_parts = [
            f"Mode: {mode}",
            "Client Only build" if CLIENT_ONLY_BUILD else "Full build",
            f"Client: {'Active' if sysmod.client_enabled else 'Stopped'}",
            f"Device preference: {device_pref.upper()}  •  CUDA ready: {'Yes' if cuda_ok else 'No'}",
            f"Model: {model_name}  •  Installed: {'Yes' if present else 'No'}",
        ]
        running = bool(sysmod.server_thread and sysmod.server_thread.is_alive())
        status_parts.append(f"Server thread: {'Running' if running else 'Not running'}")

        self.status_var.set("\n".join(status_parts))
        self.server_status_var.set(describe_server_status())
        self.cuda_status.set("CUDA runtime is available." if cuda_ok else "CUDA runtime not detected.")
        self.model_status.set("Model is installed." if present else "Model not installed.")

        if sysmod.client_enabled:
            self.start_button.state(["disabled"]); self.stop_button.state(["!disabled"])
        else:
            self.start_button.state(["!disabled"]); self.stop_button.state(["disabled"])

    # --- device actions ---
    def _apply_device(self):
        from utils.models import set_device_preference, unload_transcriber, initialize_transcriber
        choice = self.device_var.get()
        if choice not in {"cpu", "cuda", "auto"}:
            choice = "auto"
        set_device_preference(choice)

        # Offer to install CUDA if they chose GPU and it isn't ready
        if choice == "cuda" and not cuda_runtime_ready():
            if messagebox.askyesno(
                    "CUDA not ready",
                    "CUDA runtime not detected.\n\nInstall now for GPU acceleration?",
                    parent=self.window
            ):
                if install_cuda_runtime_with_progress(self.window):
                    messagebox.showinfo("CUDA", "CUDA runtime installed successfully.", parent=self.window)
                else:
                    messagebox.showwarning("CUDA", "Failed to prepare CUDA. Staying on CPU.", parent=self.window)

        # Force a reload with the new device
        unload_transcriber()
        initialize_transcriber(force=True, allow_client=True)

        self.refresh_status()

    def _install_cuda(self):
        if install_cuda_runtime_with_progress(self.window):
            messagebox.showinfo("CUDA", "CUDA runtime installed successfully.", parent=self.window)
        else:
            messagebox.showwarning("CUDA", "Failed to prepare CUDA. You can try again.", parent=self.window)
        self.refresh_status()

    # --- model actions ---
    def _apply_model(self):
        from utils.models import set_current_model_name, unload_transcriber, initialize_transcriber
        name = self.model_var.get()
        if name not in {"small", "large-v3"}:
            name = "large-v3"
        set_current_model_name(name)

        if not model_files_present(model_store_path_for(name)):
            messagebox.showinfo(
                "Model",
                "Model files are not installed yet. Use Download/Update… to fetch them before switching.",
                parent=self.window,
            )
            self.refresh_status()
            return

        # Force unload + reload the new model
        unload_transcriber()
        initialize_transcriber(force=True, allow_client=True)

        messagebox.showinfo("Model", f"Active model set to {name}.", parent=self.window)
        self.refresh_status()

    def _download_model(self):
        name = self.model_var.get()
        if name not in {"small", "large-v3"}:
            name = "large-v3"
        if download_model_with_gui(name):
            (model_store_path_for(name) / ".installed").touch(exist_ok=True)
            messagebox.showinfo("Model", "Model downloaded successfully.", parent=self.window)
        else:
            messagebox.showwarning("Model", "Model download did not complete.", parent=self.window)
        self.refresh_status()

    # --- client/server actions ---
    def start_client(self) -> None:
        start_client_listener(); self.refresh_status()

    def stop_client(self) -> None:
        stop_client_listener(); self.refresh_status()

    def refresh_servers(self) -> None:
        self.refresh_button.state(["disabled"]); self.refresh_button.config(text="Scanning…")
        self.server_status_var.set("Scanning for servers…")

        def worker() -> None:
            try:
                manual_discovery_refresh()
            finally:
                enqueue_management_task(self._on_refresh_finished)

        threading.Thread(target=worker, daemon=True).start()

    def _on_refresh_finished(self) -> None:
        if not self.is_open(): return
        self.refresh_button.state(["!disabled"]); self.refresh_button.config(text="Refresh Servers")
        self.refresh_status()

    def stop_everything(self) -> None:
        stop_client_listener(); shutdown_server()
        self.refresh_status()
        self.window.after(200, self._icon.stop)
        self.close()

    def delete_ctrlspeak(self) -> None:
        if not messagebox.askyesno("Delete CtrlSpeak",
                                   "This will remove CtrlSpeak and all local data. Continue?",
                                   parent=self.window):
            return
        self.window.after(100, lambda: initiate_self_uninstall(self._icon))

    def close(self) -> None:
        global management_window
        if self.is_open(): self.window.destroy()
        management_window = None
