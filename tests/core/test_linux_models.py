from __future__ import annotations

from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.core_headless


def test_linux_cuda_probe_uses_libcuda_without_touching_windows_loader(monkeypatch) -> None:
    from utils import cuda_probe

    calls: list[str] = []

    def cu_init(_flags: int) -> int:
        return 0

    def cu_device_get_count(pointer) -> int:
        pointer._obj.value = 1
        return 0

    driver = SimpleNamespace(
        cuInit=cu_init,
        cuDeviceGetCount=cu_device_get_count,
    )
    monkeypatch.setattr(cuda_probe.sys, "platform", "linux")
    monkeypatch.setattr(
        cuda_probe.ctypes,
        "CDLL",
        lambda name: calls.append(name) or driver,
    )
    monkeypatch.setattr(
        cuda_probe.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Windows loader must not be used on Linux")
        ),
        raising=False,
    )

    assert cuda_probe.probe_cuda_driver() is True
    assert calls == ["libcuda.so.1"]


def test_linux_cuda_probe_handles_missing_driver_library(monkeypatch) -> None:
    from utils import cuda_probe

    monkeypatch.setattr(cuda_probe.sys, "platform", "linux")
    monkeypatch.setattr(
        cuda_probe.ctypes,
        "CDLL",
        lambda _name: (_ for _ in ()).throw(OSError("not installed")),
    )

    assert cuda_probe.probe_cuda_driver() is False


def test_automatic_cuda_runtime_installer_remains_windows_only() -> None:
    from utils.cuda_probe import automatic_runtime_install_supported

    assert automatic_runtime_install_supported("linux") is False
    assert automatic_runtime_install_supported("win32") is True
