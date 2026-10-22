from __future__ import annotations

import asyncio
from pathlib import Path
import threading
import time
import sys

import pytest

from utils.cancellable_http import TransportFailure, upload_audio, post_audio
from utils.recording_recovery import RecordingRecovery
from utils.request_control import current_request_control, request_control
from utils.transcription_backend import _transcription_http_error
from utils.ui_state import TranscriptionUiSession

pytestmark = pytest.mark.core_headless


def test_recovery_is_one_memory_only_clip_with_expiry_and_discard(tmp_path):
    clock = [10.0]
    store = RecordingRecovery(ttl_seconds=60, clock=lambda: clock[0])
    path = tmp_path / "audio.wav"
    path.write_bytes(b"a" * 200)
    initial_files = set(tmp_path.iterdir())
    try:
        assert store.retain(path)
        assert set(tmp_path.iterdir()) == initial_files
        old_buffer = store._audio
        path.write_bytes(b"b" * 300)
        assert store.retain(path)
        assert old_buffer == bytearray(200)
        target = tmp_path / "retry.wav"
        assert store.take(lambda: target) == target
        assert target.read_bytes() == b"b" * 300
        assert store.seconds_remaining() == 0
        assert store.take(lambda: tmp_path / "unwanted.wav") is None
        assert store.retain(path)
        clock[0] += 61
        assert store.seconds_remaining() == 0
        assert store._audio is None
    finally:
        store.discard()


def test_recovery_never_deletes_collision_and_retains_retry_audio(tmp_path):
    path = tmp_path / "input.wav"
    path.write_bytes(b"a" * 100)
    store = RecordingRecovery()
    try:
        assert store.retain(path)
        with pytest.raises(FileExistsError):
            store.take(lambda: path)
        assert path.read_bytes() == b"a" * 100
        assert store.seconds_remaining() > 0
    finally:
        store.discard()


def test_recovery_old_timer_cannot_expire_new_clip(tmp_path):
    path = tmp_path / "input.wav"
    path.write_bytes(b"a" * 100)
    store = RecordingRecovery()
    try:
        assert store.retain(path)
        old_callback = store._timer.function
        assert store.retain(path)
        old_callback()
        assert store.seconds_remaining() > 0
        store.discard()
        assert store._audio is None
    finally:
        store.discard()


def test_recovery_bound_rejects_oversize_audio(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"x" * 101)
    store = RecordingRecovery(max_bytes=100)
    assert not store.retain(path)
    assert store.seconds_remaining() == 0


def test_request_context_is_restored_and_thread_local():
    first, second = threading.Event(), threading.Event()
    assert current_request_control() is None
    with request_control(first):
        assert current_request_control().cancel is first
        with request_control(second):
            assert current_request_control().cancel is second
        assert current_request_control().cancel is first
    assert current_request_control() is None


@pytest.mark.parametrize("status,payload,category", [
    (401, {"detail": "secret upstream body"}, "gateway_auth_failed"),
    (403, {}, "gateway_auth_failed"),
    (401, {"detail": {"category": "openai_invalid_key", "message": "sk-secret"}}, "openai_invalid_key"),
    (402, {"detail": {"category": "openai_quota_exhausted"}}, "openai_quota_exhausted"),
    (503, {}, "providers_exhausted"), (429, {}, "gateway_busy"),
])
def test_http_errors_keep_gateway_and_provider_auth_distinct(status, payload, category):
    error = _transcription_http_error(status, payload)
    assert error.category == category
    assert "secret" not in str(error)


def test_processing_stages_are_distinct_without_changing_busy_phase():
    session = TranscriptionUiSession()
    session.begin_recording()
    session.begin_processing()
    for stage, headline in (("preparing", "Preparing audio"), ("uploading", "Uploading"),
                            ("transcribing", "Transcribing"), ("inserting", "Inserting text")):
        session.set_processing_stage(stage)
        assert session.snapshot().headline == headline
        assert session.snapshot().can_cancel
    session.fail("gateway_auth_failed")
    assert "bearer token" in session.snapshot().detail


@pytest.mark.parametrize("mode", ["cancel", "deadline"])
def test_http_cancel_and_deadline_close_socket_and_reap_tasks(tmp_path, mode):
    pytest.importorskip("aiohttp")
    path = tmp_path / "audio.flac"
    path.write_bytes(b"fLaC" + b"x" * 10000)

    async def scenario():
        connected = asyncio.Event()
        disconnected = asyncio.Event()
        cancel = threading.Event()
        handlers = set()

        async def handle(reader, writer):
            task = asyncio.current_task()
            handlers.add(task)
            connected.set()
            try:
                while await reader.read(65536):
                    pass  # Read upload but deliberately never send a response.
                disconnected.set()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(task)

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        pending = asyncio.create_task(upload_audio(
            f"http://127.0.0.1:{port}/v1/transcribe", path, filename="recording.flac",
            content_type="audio/flac", headers={}, fields={}, cancel_check=cancel.is_set,
            progress=None, total_seconds=0.20 if mode == "deadline" else 3.0,
        ))
        try:
            await asyncio.wait_for(connected.wait(), 1)
            started = time.monotonic()
            if mode == "cancel":
                cancel.set()
            with pytest.raises(TransportFailure) as caught:
                await asyncio.wait_for(pending, 2)
            assert caught.value.category == ("cancelled" if mode == "cancel" else "request_timeout")
            assert time.monotonic() - started < 1.0
            await asyncio.wait_for(disconnected.wait(), 1)
            await asyncio.sleep(0)
            assert pending.done()
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            server.close()
            await server.wait_closed()
            for handler in tuple(handlers):
                handler.cancel()
            await asyncio.gather(*tuple(handlers), return_exceptions=True)

    asyncio.run(asyncio.wait_for(scenario(), 5))


def test_http_precancel_does_not_open_connection(tmp_path):
    path = tmp_path / "audio.flac"
    path.write_bytes(b"fLaC")
    with pytest.raises(TransportFailure) as caught:
        asyncio.run(upload_audio("http://127.0.0.1:1/", path, filename="a.flac", content_type="audio/flac",
                                 headers={}, fields={}, cancel_check=lambda: True, progress=None))
    assert caught.value.category == "cancelled"


def test_hostname_dns_timeout_closes_owned_resolver(monkeypatch, tmp_path):
    import aiohttp
    closed = []

    class StalledResolver:
        async def resolve(self, *_args, **_kwargs):
            await asyncio.sleep(20)

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(aiohttp, "AsyncResolver", StalledResolver)
    path = tmp_path / "audio.flac"
    path.write_bytes(b"fLaC")

    async def scenario():
        with pytest.raises(TransportFailure) as error:
            await upload_audio("http://stalled.example.invalid/", path, filename="a.flac", content_type="audio/flac",
                               headers={}, fields={}, cancel_check=lambda: False, progress=None,
                               total_seconds=0.3, connect_seconds=0.1)
        assert error.value.category == "gateway_unreachable"
        assert closed == [True]

    asyncio.run(asyncio.wait_for(scenario(), 2))


def test_http_streams_lossless_body_and_reports_actual_stages(tmp_path):
    from aiohttp import web
    path = tmp_path / "audio.flac"
    original = b"fLaC" + b"lossless-audio" * 100000
    path.write_bytes(original)
    stages = []
    client_loops = []

    def progress(stage):
        stages.append(stage)
        client_loops.append(type(asyncio.get_running_loop()).__name__)

    async def scenario():
        async def handle(request):
            fields = await request.post()
            assert fields["audio"].file.read() == original
            assert fields["strategy"] == "ubuntu-gpu-preferred"
            assert request.headers["X-CtrlSpeak-Request-ID"] == "safe-request-id"
            return web.json_response({"text": "worked", "id": "tx", "raw_text": "worked"})

        app = web.Application(client_max_size=4 * 1024 * 1024)
        app.router.add_post("/v1/transcribe", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            # Exercise the synchronous entrypoint actually used by desktop,
            # not only upload_audio on pytest's outer event loop. Windows'
            # native Proactor transport is required on the real gateway route.
            status, payload, timings = await asyncio.to_thread(
                post_audio,
                f"http://127.0.0.1:{port}/v1/transcribe", path, filename="recording.flac",
                content_type="audio/flac", fields={"strategy": "ubuntu-gpu-preferred"},
                headers={"X-CtrlSpeak-Request-ID": "safe-request-id"},
                cancel_check=lambda: False, progress=progress,
            )
            assert status == 200 and payload["text"] == "worked"
            assert stages == ["uploading", "transcribing"]
            if sys.platform == "win32":
                assert client_loops == ["ProactorEventLoop", "ProactorEventLoop"]
            assert timings["request_ms"] >= timings["upload_ms"] >= 0
        finally:
            await runner.cleanup()

    asyncio.run(asyncio.wait_for(scenario(), 5))
