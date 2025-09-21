# -*- coding: utf-8 -*-
"""
ensure_runtime.py
Minimal, AppData-aware CUDA runtime bootstrap for CtrlSpeak.

- Uses utils.config_paths.get_config_dir() -> %APPDATA%/CtrlSpeak
- Downloads *DLL bundles* (zip files of needed CUDA/cuDNN/cuBLAS DLLs)
- Extracts into:
    %APPDATA%/CtrlSpeak/cuda/bin
    %APPDATA%/CtrlSpeak/cuda/cudnn/bin
    %APPDATA%/CtrlSpeak/cuda/cublas/bin

Note: We avoid full CUDA installers. We only fetch the redistributable DLL sets
that the app needs at runtime. Provide direct ZIP URLs for your chosen versions.
"""

from __future__ import annotations
import os, sys, platform, subprocess, shutil, zipfile, io, urllib.request, time
from pathlib import Path
from typing import Optional, Tuple

from .downloader import ensure_file
from .config_paths import get_config_dir

# --- Configure your redistributable ZIP URLs here ---
# Each should be a zip containing the appropriate DLLs under a 'bin/' folder.
CUDA_DLL_BUNDLE_URL: Optional[str]   = os.environ.get("CTRLSPEAK_CUDA_ZIP_URL")
CUDNN_DLL_BUNDLE_URL: Optional[str]  = os.environ.get("CTRLSPEAK_CUDNN_ZIP_URL")
CUBLAS_DLL_BUNDLE_URL: Optional[str] = os.environ.get("CTRLSPEAK_CUBLAS_ZIP_URL")

def _extract_zip_to(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(dest_dir))

def _ensure_zip(url: str, dest_zip: Path, label: str) -> Path:
    return ensure_file(url, dest_zip, label)

def ensure_cuda_runtime() -> None:
    """Ensure CUDA-related DLLs live under %APPDATA%/CtrlSpeak/cuda/*/bin."""
    if not sys.platform.startswith("win"):
        return

    cfg = get_config_dir()
    cuda_root = cfg / "cuda"
    cuda_bin    = cuda_root / "bin"
    cudnn_bin   = cuda_root / "cudnn" / "bin"
    cublas_bin  = cuda_root / "cublas" / "bin"
    for d in (cuda_bin, cudnn_bin, cublas_bin):
        d.mkdir(parents=True, exist_ok=True)

    # If DLLs already present, bail out fast
    if any(cuda_bin.glob("*.dll")) and any(cudnn_bin.glob("*.dll")) and any(cublas_bin.glob("*.dll")):
        return

    downloads_dir = cfg / "temp"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    if CUDA_DLL_BUNDLE_URL:
        zip_path = downloads_dir / "cuda_runtime_dlls.zip"
        _ensure_zip(CUDA_DLL_BUNDLE_URL, zip_path, "CUDA Runtime DLLs")
        _extract_zip_to(zip_path, cuda_bin.parent)  # expects bin/ inside
    if CUDNN_DLL_BUNDLE_URL:
        zip_path = downloads_dir / "cudnn_dlls.zip"
        _ensure_zip(CUDNN_DLL_BUNDLE_URL, zip_path, "cuDNN DLLs")
        _extract_zip_to(zip_path, cudnn_bin.parent)
    if CUBLAS_DLL_BUNDLE_URL:
        zip_path = downloads_dir / "cublas_dlls.zip"
        _ensure_zip(CUBLAS_DLL_BUNDLE_URL, zip_path, "cuBLAS DLLs")
        _extract_zip_to(zip_path, cublas_bin.parent)

    # (Removed: os.startfile(cuda_root))
