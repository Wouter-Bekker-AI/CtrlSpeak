"""An owned, cancellable HTTP operation: no detached request thread to outlive it.

Cancellation closes the client connection; it cannot undo provider work or
billing already accepted by the gateway. A retry is always a new, explicit act.
"""
from __future__ import annotations

import asyncio
import json
import ipaddress
from pathlib import Path
import sys
import time
from typing import Callable, Mapping
from urllib.parse import urlsplit


class TransportFailure(RuntimeError):
    def __init__(self, category: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.category = category
        self.retryable = retryable


async def upload_audio(
    url: str, path: Path, *, filename: str, content_type: str,
    headers: Mapping[str, str], fields: Mapping[str, str],
    cancel_check: Callable[[], bool], progress: Callable[[str], None] | None,
    total_seconds: float = 180.0, connect_seconds: float = 3.0,
    read_seconds: float = 90.0,
) -> tuple[int, dict, dict[str, float]]:
    import aiohttp

    started = time.monotonic()
    timings: dict[str, float] = {}

    def stage(name: str):
        if progress:
            progress(name)

    async def body():
        stage("uploading")
        # An async iterable keeps file reads bounded and avoids an executor
        # worker which could survive request cancellation. Local 64KiB reads
        # are short; the HTTP writer supplies network backpressure.
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        timings["upload_ms"] = round((time.monotonic() - started) * 1000, 2)
        stage("transcribing")

    async def send():
        form = aiohttp.FormData()
        for name, value in fields.items():
            form.add_field(name, value)
        form.add_field("audio", body(), filename=filename, content_type=content_type)
        timeout = aiohttp.ClientTimeout(total=total_seconds, connect=connect_seconds,
                                       sock_connect=connect_seconds, sock_read=read_seconds)
        class NumericResolver:
            async def resolve(self, *_args, **_kwargs):
                raise RuntimeError("Literal IP route must not perform DNS resolution")

            async def close(self):
                return None

        resolver = NumericResolver()
        try:
            ipaddress.ip_address(urlsplit(url).hostname or "")
        except ValueError:
            # c-ares DNS has no getaddrinfo executor thread for loop shutdown
            # to wait on after a timed-out/cancelled hostname resolution.
            resolver = aiohttp.AsyncResolver()
        connector = aiohttp.TCPConnector(resolver=resolver)
        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
                # Never redirect credential-bearing uploads to another host.
                async with session.post(url, data=form, headers=headers, allow_redirects=False) as response:
                    data = bytearray()
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        data.extend(chunk)
                        if len(data) > 2 * 1024 * 1024:
                            raise TransportFailure("invalid_response", "Gateway response exceeded the safety limit.")
                    try:
                        payload = json.loads(data)
                    except (ValueError, UnicodeError) as exc:
                        if response.status < 200 or response.status >= 300:
                            return response.status, {}, timings
                        raise TransportFailure("invalid_response", "Gateway returned an invalid JSON response.") from exc
                    if not isinstance(payload, dict):
                        raise TransportFailure("invalid_response", "Gateway returned an invalid response shape.")
                    timings["request_ms"] = round((time.monotonic() - started) * 1000, 2)
                    return response.status, payload, timings
        finally:
            await resolver.close()

    async def watch_cancel():
        while not cancel_check():
            await asyncio.sleep(0.05)

    if cancel_check():
        raise TransportFailure("cancelled", "Transcription cancelled.")
    request = asyncio.create_task(send())
    watcher = asyncio.create_task(watch_cancel())
    try:
        done, _ = await asyncio.wait({request, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if watcher in done or cancel_check():
            raise TransportFailure("cancelled", "Transcription cancelled. Provider work already accepted may still be charged.")
        return await request
    except aiohttp.ConnectionTimeoutError as exc:
        raise TransportFailure("gateway_unreachable", "Cannot connect to the gateway. Check the VPN, address and gateway service.", retryable=True) from exc
    except asyncio.TimeoutError as exc:
        raise TransportFailure("request_timeout", "The transcription request exceeded its deadline. A manual retry may incur another provider charge.", retryable=True) from exc
    except aiohttp.ClientConnectorCertificateError as exc:
        raise TransportFailure("gateway_tls_error", "The gateway TLS certificate could not be verified.") from exc
    except aiohttp.ClientError as exc:
        raise TransportFailure("gateway_unreachable", "The gateway connection failed. Check your VPN and gateway service.", retryable=True) from exc
    finally:
        # Await both tasks after cancelling: return implies socket/session closed.
        for task in (request, watcher):
            if not task.done():
                task.cancel()
        await asyncio.gather(request, watcher, return_exceptions=True)


def _run_owned_loop(operation):
    # aiodns needs a selector on Windows. A private loop avoids changing the
    # process policy, the GUI loop, or another request's cancellation state.
    loop = asyncio.SelectorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
    try:
        return loop.run_until_complete(operation)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def post_audio(*args, **kwargs):
    return _run_owned_loop(upload_audio(*args, **kwargs))


def request_transport_self_test() -> dict:
    """Package health check for binary dependencies; never connects to a host."""
    async def check():
        import aiohttp
        import aiodns
        import pycares
        resolver = aiohttp.AsyncResolver()
        connector = aiohttp.TCPConnector(resolver=resolver)
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                form = aiohttp.FormData()
                form.add_field("audio", b"fLaC", filename="health.flac", content_type="audio/flac")
                assert form().content_type.startswith("multipart/form-data;")
                assert not session.closed
            return {"ok": True, "cancellable_http": True, "async_dns": True,
                    "aiohttp_version": aiohttp.__version__}
        finally:
            await resolver.close()
    return _run_owned_loop(check())
