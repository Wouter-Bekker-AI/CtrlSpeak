# utils/net_discovery.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import socket
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from utils.config_paths import settings, settings_lock, save_settings, get_logger

logger = get_logger(__name__)

DISCOVERY_INTERVAL_SECONDS = 5.0
DISCOVERY_ENTRY_TTL = 15.0
SERVER_BROADCAST_SIGNATURE = "CTRLSPEAK_SERVER"

@dataclass
class ServerInfo:
    host: str
    port: int
    last_seen: float

class DiscoveryListener(threading.Thread):
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        self.stop_event = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(("", port))
        except OSError:
            self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(1.0)
        self.registry: Dict[Tuple[str, int], ServerInfo] = {}

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                self._prune(); continue
            except OSError:
                break
            try:
                message = data.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            parts = message.split("|")
            if len(parts) != 3 or parts[0] != SERVER_BROADCAST_SIGNATURE:
                continue
            try:
                host = parts[1]; port = int(parts[2])
            except ValueError:
                continue
            key = (host, port)
            self.registry[key] = ServerInfo(host=host, port=port, last_seen=time.time())

    def _prune(self) -> None:
        now = time.time()
        expired = [key for key, info in self.registry.items()
                   if now - info.last_seen > DISCOVERY_ENTRY_TTL]
        for key in expired:
            self.registry.pop(key, None)

    def get_best_server(self) -> Optional[ServerInfo]:
        self._prune()
        if not self.registry:
            return None
        return max(self.registry.values(), key=lambda entry: entry.last_seen)

    def clear_registry(self) -> None:
        self.registry.clear()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.sock.close()
        except Exception:
            logger.exception("Failed to close discovery listener socket")


def get_preferred_server_settings() -> tuple[Optional[str], Optional[int]]:
    with settings_lock:
        host = settings.get("preferred_server_host")
        port = settings.get("preferred_server_port")
    if isinstance(host, str):
        host = host.strip() or None
    if isinstance(port, str):
        try:
            port = int(port)
        except ValueError:
            port = None
    if host is None or port is None:
        return None, None
    return host, int(port)


def set_preferred_server(host: str, port: int) -> None:
    with settings_lock:
        settings["preferred_server_host"] = host
        settings["preferred_server_port"] = int(port)
    save_settings()


def clear_preferred_server() -> None:
    with settings_lock:
        settings["preferred_server_host"] = None
        settings["preferred_server_port"] = None
    save_settings()


def parse_server_target(value: str) -> tuple[str, int]:
    target = value.strip()
    if not target:
        raise ValueError("Server address cannot be empty.")
    if target.count(":") == 0:
        host = target
        port = int(65432)
    else:
        host, port_str = target.rsplit(":", 1)
        host = host.strip()
        if not host:
            raise ValueError("Server host cannot be empty.")
        try:
            port = int(port_str.strip())
        except ValueError as exc:
            raise ValueError("Port must be a number.") from exc
    if port <= 0 or port > 65535:
        raise ValueError("Port must be between 1 and 65535.")
    return host, port


def probe_server(host: str, port: int, timeout: float = 2.0) -> bool:
    import http.client
    conn: Optional[http.client.HTTPConnection] = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/ping")
        response = conn.getresponse(); response.read()
        return response.status == 200
    except Exception:
        logger.exception("Failed to probe server %s:%s", host, port)
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                logger.exception("Failed to close probe connection to %s:%s", host, port)


def register_manual_server(host: str, port: int, update_preference: bool = True) -> ServerInfo:
    server_info = ServerInfo(host=host, port=int(port), last_seen=time.time())
    if update_preference:
        set_preferred_server(server_info.host, server_info.port)
    return server_info


def ensure_preferred_server_registered(probe: bool = False) -> Optional[ServerInfo]:
    host, port = get_preferred_server_settings()
    if host is None or port is None:
        return None
    if probe and not probe_server(host, port):
        return None
    # CHANGED: use register_manual_server so this function isn't duplicating object creation
    # and so there's a single code path to materialize a ServerInfo + (optionally) persist prefs.
    return register_manual_server(host, port, update_preference=False)  # no pref change here


def send_discovery_query(timeout: float = 1.0) -> None:
    """Broadcast a discovery query on the configured discovery port."""
    with settings_lock:
        port = int(settings.get("discovery_port", 54330))
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout)
        message = f"{SERVER_BROADCAST_SIGNATURE}_QUERY".encode("utf-8")
        sock.sendto(message, ("255.255.255.255", port))
    except Exception:
        logger.exception("Failed to broadcast discovery query on port %s", port)
    finally:
        try:
            sock.close()
        except Exception:
            logger.exception("Failed to close discovery query socket")


def manual_discovery_refresh(discovery_listener=None, wait_time: float = 1.5) -> Optional[ServerInfo]:
    if discovery_listener is None:
        return ensure_preferred_server_registered(probe=True)
    discovery_listener.clear_registry()
    # Use the restored helper to broadcast the query
    send_discovery_query()
    time.sleep(max(wait_time, 0.5))
    server = discovery_listener.get_best_server()
    if server:
        if server.host not in {"local", "local-cpu"}:
            set_preferred_server(server.host, server.port)
        # CHANGED (optional normalization): still return the discovered server as-is.
        return server
    # Fallback will now flow through register_manual_server via ensure_preferred_server_registered
    return ensure_preferred_server_registered(probe=True)


def get_best_server(discovery_listener=None) -> Optional[ServerInfo]:
    if discovery_listener is None:
        return ensure_preferred_server_registered(probe=True)
    server = discovery_listener.get_best_server()
    if server:
        return server
    return ensure_preferred_server_registered(probe=True)


def get_advertised_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tmp:
            tmp.connect(("8.8.8.8", 80))
            return tmp.getsockname()[0]
    except Exception:
        logger.exception("Failed to determine advertised host IP")
        return "127.0.0.1"


def manage_discovery_broadcast(stop_event: threading.Event, port: int, server_port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        while not stop_event.is_set():
            host_ip = get_advertised_host_ip()
            message = f"{SERVER_BROADCAST_SIGNATURE}|{host_ip}|{server_port}".encode("utf-8")
            try:
                sock.sendto(message, ("255.255.255.255", port))
            except Exception:
                logger.exception("Failed to send discovery broadcast from %s:%s", host_ip, port)
            stop_event.wait(DISCOVERY_INTERVAL_SECONDS)
    finally:
        sock.close()


def listen_for_discovery_queries(stop_event: threading.Event, port: int, server_port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except OSError as exc:
        logger.warning("Failed to enable SO_REUSEADDR on discovery query listener: %s", exc)
    try:
        sock.bind(("", port)); sock.settimeout(1.0)
    except OSError:
        sock.close(); return
    try:
        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                message = data.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if message != f"{SERVER_BROADCAST_SIGNATURE}_QUERY":
                continue
            host_ip = get_advertised_host_ip()
            response = f"{SERVER_BROADCAST_SIGNATURE}|{host_ip}|{server_port}".encode("utf-8")
            try:
                sock.sendto(response, addr)
            except Exception:
                logger.exception("Failed to send discovery response to %s", addr)
    finally:
        try:
            sock.close()
        except Exception:
            logger.exception("Failed to close discovery query listener socket")
