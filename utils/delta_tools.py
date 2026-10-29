"""Pinned native VCDIFF tool provenance; no network activity at app runtime."""
from pathlib import Path
import hashlib
import os
import sys

TOOL_VERSION = "3.2.2"
TOOL_ROOT = "https://github.com/jmacd/xdelta/releases/download/v3.2.2/"
LICENSE_URL = "https://raw.githubusercontent.com/jmacd/xdelta/v3.2.2/xdelta3/LICENSE"
LICENSE_SHA256 = "6cf63d87586c7c25c3ab8e62eef5c75bbaa982b0a6f9c00c59e2720255aac9ec"
TOOLS = {
    "windows": {"archive": "xdelta3-3.2.2-windows-x86_64.zip",
                "archive_sha256": "d48f7651d86b78fdca397057b46601f89131f37219675df736db70af44ec3379",
                "binary": "xdelta3.exe",
                "binary_sha256": "732980d52f9220dc81817ccdd0c295eadce8a0048a24c2a771a73f791ac824e9"},
    "linux": {"archive": "xdelta3-3.2.2-linux-x86_64.tar.gz",
              "archive_sha256": "a8ce7d2c943ae2b3d308942643cb36d4805a1f16775abe1be64fee292d7c2506",
              "binary": "xdelta3",
              "binary_sha256": "709508773841aa1e96d39e86b143c4aee496be8eb42a3185bf42fc480dc26ebd"},
}


def tool_path(platform_name=None):
    platform_name = platform_name or ("windows" if sys.platform.startswith("win") else "linux")
    spec = TOOLS[platform_name]
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "tools" / "xdelta3" / spec["binary"]
    return Path(__file__).resolve().parents[1] / ".codex-tools" / "xdelta3" / platform_name / spec["binary"]


def verified_tool(platform_name=None):
    platform_name = platform_name or ("windows" if sys.platform.startswith("win") else "linux")
    path = tool_path(platform_name)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != TOOLS[platform_name]["binary_sha256"]:
        raise RuntimeError("The bundled differential-update tool is unavailable or invalid")
    if platform_name == "linux" and not os.access(path, os.X_OK):
        # Bundle as data to prevent PyInstaller rewriting ELF/RPATH bytes.
        # One-file extraction may omit executable mode; restore owner-execute
        # only after checking the pinned content hash, never modifying bytes.
        path.chmod(path.stat().st_mode | 0o100)
    return path
