"""Small native CUDA-driver probe shared by Linux and Windows runtimes."""
from __future__ import annotations

import ctypes
import sys
from typing import Any

from utils.config_paths import get_logger


logger = get_logger(__name__)


def automatic_runtime_install_supported(platform_name: str | None = None) -> bool:
    """The bundled wheel extractor is currently maintained only for Windows."""
    return (platform_name or sys.platform).lower().startswith("win")


def _load_cuda_driver() -> Any | None:
    if sys.platform.startswith("win"):
        try:
            return ctypes.windll.nvcuda  # type: ignore[attr-defined]
        except Exception:
            try:
                ctypes.WinDLL("nvcuda.dll")
                return ctypes.windll.nvcuda  # type: ignore[attr-defined]
            except Exception:
                logger.debug("CUDA driver DLL nvcuda.dll was not found", exc_info=True)
                return None
    if sys.platform.startswith("linux"):
        try:
            return ctypes.CDLL("libcuda.so.1")
        except OSError:
            logger.debug("CUDA driver library libcuda.so.1 was not found", exc_info=True)
            return None
    return None


def probe_cuda_driver() -> bool:
    driver = _load_cuda_driver()
    if driver is None:
        return False
    try:
        cu_init = driver.cuInit
        cu_init.argtypes = [ctypes.c_uint]
        cu_init.restype = ctypes.c_int
        result = cu_init(0)
    except Exception:
        logger.debug("Failed to invoke cuInit while probing CUDA hardware", exc_info=True)
        return False
    if result == 100:  # CUDA_ERROR_NO_DEVICE
        return False
    if result != 0:
        logger.debug("cuInit returned error code %s", result)
        return False
    try:
        cu_get_count = driver.cuDeviceGetCount
        cu_get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cu_get_count.restype = ctypes.c_int
        count = ctypes.c_int(0)
        status = cu_get_count(ctypes.byref(count))
    except Exception:
        logger.debug("Failed to query CUDA device count", exc_info=True)
        return False
    if status != 0:
        logger.debug("cuDeviceGetCount returned error code %s", status)
        return False
    return count.value > 0
