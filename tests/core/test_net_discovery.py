from __future__ import annotations

import pytest

from utils import config_paths
from utils import net_discovery

pytestmark = pytest.mark.core_headless


def test_parse_server_target_default_port():
    assert net_discovery.parse_server_target("localhost") == ("localhost", 65432)


def test_parse_server_target_invalid_port():
    with pytest.raises(ValueError):
        net_discovery.parse_server_target("localhost:not-a-port")


def test_ensure_preferred_server_registered(monkeypatch):
    config_paths.load_settings()
    net_discovery.set_preferred_server("127.0.0.1", 6000)
    monkeypatch.setattr(net_discovery, "probe_server", lambda *a, **k: True)
    server = net_discovery.ensure_preferred_server_registered(probe=True)
    assert server is not None
    assert server.host == "127.0.0.1"
    assert server.port == 6000
