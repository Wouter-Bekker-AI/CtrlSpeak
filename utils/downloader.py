# -*- coding: utf-8 -*-
"""
downloader.py
A small, reusable download manager with a Tkinter progress window, speed, ETA,
and a simple prompt API. This is designed to be used at app startup to pull
CUDA/cuDNN runtimes (on Windows) and model weights as-needed.
"""

from __future__ import annotations
import os, sys, time, math, threading, queue, urllib.request
from pathlib import Path
from typing import Optional, Callable
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    tk = None
    ttk = None
    messagebox = None

_CHUNK = 1 << 15  # 32 KiB

class DownloadItem:
    def __init__(self, url: str, dest: Path, label: str):
        self.url = url
        self.dest = Path(dest)
        self.label = label
        self.total = None
        self.downloaded = 0
        self.start_ts = None
        self.done = False
        self.error: Optional[Exception] = None

    @property
    def speed_bps(self) -> Optional[float]:
        if self.start_ts is None: return None
        dt = max(time.time() - self.start_ts, 1e-6)
        return self.downloaded / dt

    @property
    def eta_s(self) -> Optional[float]:
        if self.total is None or self.speed_bps is None or self.speed_bps <= 0:
            return None
        remain = max(self.total - self.downloaded, 0)
        return remain / self.speed_bps

def human_bytes(n: Optional[float]) -> str:
    if n is None: return "unknown"
    units = ["B","KB","MB","GB","TB"]
    i = 0
    n = float(n)
    while n >= 1024 and i < len(units)-1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {units[i]}"

def human_time(s: Optional[float]) -> str:
    if s is None: return "—"
    s = int(s)
    if s < 60: return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60: return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"

def _download_worker(item: DownloadItem, progress_cb: Optional[Callable[[DownloadItem], None]] = None):
    try:
        item.start_ts = time.time()
        req = urllib.request.Request(item.url, headers={"User-Agent": "CtrlSpeak/1.0"})
        with urllib.request.urlopen(req) as r:
            total = r.headers.get("Content-Length")
            item.total = int(total) if total is not None and total.isdigit() else None
            item.dest.parent.mkdir(parents=True, exist_ok=True)
            with open(item.dest, "wb") as f:
                while True:
                    chunk = r.read(_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    item.downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(item)
        item.done = True
        if progress_cb:
            progress_cb(item)
    except Exception as e:
        item.error = e
        if progress_cb:
            progress_cb(item)

class DownloadUI:
    def __init__(self, title="Downloading components"):
        if tk is None:
            raise RuntimeError("Tkinter not available; cannot show download UI.")
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("580x160")
        self.root.resizable(False, False)

        self.var_label = tk.StringVar(value="Preparing...")
        self.var_progress = tk.DoubleVar(value=0.0)
        self.var_speed = tk.StringVar(value="Speed: —")
        self.var_eta = tk.StringVar(value="ETA: —")

        pad = {"padx": 12, "pady": 6}
        tk.Label(self.root, textvariable=self.var_label, anchor="w").pack(fill="x", **pad)
        self.bar = ttk.Progressbar(self.root, maximum=100.0, variable=self.var_progress)
        self.bar.pack(fill="x", **pad)
        row = tk.Frame(self.root); row.pack(fill="x", **pad)
        tk.Label(row, textvariable=self.var_speed, anchor="w").pack(side="left")
        tk.Label(row, textvariable=self.var_eta, anchor="right").pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._closed = False

    def _on_close(self):
        # Prevent closing during download to avoid partial files
        if messagebox:
            messagebox.showinfo("Please wait", "Download in progress; please wait until it completes.")
        return

    def update_from_item(self, item: DownloadItem):
        label = f"{item.label} — {human_bytes(item.downloaded)}"
        if item.total:
            label += f" / {human_bytes(item.total)}"
            pct = (item.downloaded / item.total) * 100.0
        else:
            pct = 0.0 if item.downloaded == 0 else 50.0  # indeterminate-ish

        self.var_label.set(label)
        self.var_progress.set(pct)
        self.var_speed.set(f"Speed: {human_bytes(item.speed_bps)}/s" if item.speed_bps else "Speed: —")
        self.var_eta.set(f"ETA: {human_time(item.eta_s)}")

        self.root.update_idletasks()
        self.root.update()

def download_with_ui(url: str, dest: Path, label: str = "Downloading file") -> Path:
    """
    Downloads url -> dest with a Tkinter progress window and returns dest.
    Raises on error.
    """
    item = DownloadItem(url, dest, label)
    ui = DownloadUI(title=label)

    def progress_cb(_item: DownloadItem):
        ui.update_from_item(_item)

    t = threading.Thread(target=_download_worker, args=(item, progress_cb), daemon=True)
    t.start()
    while not item.done and item.error is None:
        time.sleep(0.05)
        ui.root.update_idletasks()
        ui.root.update()
    if item.error:
        raise item.error
    ui.root.destroy()
    return dest

def ensure_file(url: str, dest: Path, label: str) -> Path:
    """Check if dest exists; if not, prompt user and download with UI."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    # ask user
    if tk is not None and messagebox is not None:
        root = tk.Tk(); root.withdraw()
        resp = messagebox.askyesno("Download required", f"{label} is missing.\nDownload now?\n\n{url}")
        root.destroy()
        if not resp:
            raise RuntimeError(f"User declined download for {label}")
    return download_with_ui(url, dest, label)