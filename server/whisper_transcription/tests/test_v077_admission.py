from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from app.admission import Admission, WorkOwner, safe_timings
from app.audio_transport import AudioLimitError
from app.corrections import CorrectionStore
from app.main import create_app
from app.providers import LocalWhisperProvider, ProviderContext, ProviderFailure, ProviderRouter, RemoteWorkerProvider


class BlockingBackend:
    name, device, compute_type = "large-v3-turbo", "cuda", "float16"

    def __init__(self, error=None):
        self.entered, self.release = threading.Event(), threading.Event()
        self.error = error
        self.path = None

    def load(self):
        pass

    def transcribe(self, path, *_args):
        self.path = path
        self.entered.set()
        assert self.release.wait(3), "test must release the inference thread"
        assert path.exists(), "audio must remain until real inference completes"
        if self.error:
            raise self.error
        return {"raw_text": "private transcript", "language": "en", "segments": []}


def upload():
    return {"files": {"audio": ("private-filename.wav", b"RIFF-test", "audio/wav")}}


async def wait_finished(app):
    async with asyncio.timeout(2):
        while app.state.owned_tasks:
            await asyncio.sleep(0.005)


def test_saturated_worker_fails_before_consuming_another_upload(tmp_path):
    async def run():
        backend = BlockingBackend()
        app = create_app(backend=backend, service_role="worker", environ={},
                         temp_dir=tmp_path / "uploads")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
                first = asyncio.create_task(client.post("/v1/worker/transcribe", **upload()))
                try:
                    assert await asyncio.to_thread(backend.entered.wait, 1)
                    async def unread_body():
                        pytest.fail("busy request body must not be read")
                        yield b"never"
                    busy = await client.post("/v1/worker/transcribe", content=unread_body())
                    assert busy.status_code == 503
                    assert busy.json()["detail"]["category"] == "worker_busy"
                    health = (await client.get("/health")).json()
                    assert health["status"] == "ready"
                    assert health["admission"]["status"] == "busy"
                    assert health["admission"]["queued"] == 0
                    assert health["admission"]["available"] == 0
                    caps = (await client.get("/v1/capabilities")).json()
                    assert caps["provider"]["status"] == "busy"
                finally:
                    backend.release.set()
                    response = await first
                assert response.status_code == 200
                assert app.state.admission.snapshot()["active"] == 0
                assert not list((tmp_path / "uploads").iterdir())
    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_deadline_or_caller_cancellation_never_frees_running_inference(tmp_path, cancel):
    async def run():
        backend = BlockingBackend()
        app = create_app(backend=backend, service_role="worker",
                         environ={"CTRLSPEAK_REQUEST_DEADLINE_SECONDS": "0.1" if not cancel else "2"},
                         temp_dir=tmp_path / "uploads")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
                request = asyncio.create_task(client.post("/v1/worker/transcribe", **upload()))
                try:
                    assert await asyncio.to_thread(backend.entered.wait, 1)
                    if cancel:
                        request.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await request
                    else:
                        response = await request
                        assert response.status_code == 504
                        assert response.json()["detail"]["category"] == "worker_deadline_exceeded"
                    assert app.state.admission.snapshot()["active"] == 1
                    assert backend.path.exists()
                    rejected = await client.post("/v1/worker/transcribe", **upload())
                    assert rejected.status_code == 503
                finally:
                    backend.release.set()
                    await wait_finished(app)
                    if not request.done():
                        await request
                assert app.state.admission.snapshot()["active"] == 0
                assert not backend.path.exists()
    asyncio.run(run())


@pytest.mark.parametrize("error,status,category", [
    (RuntimeError("private-error-with-token"), 500, "worker_error"),
    (AudioLimitError("private-filename"), 422, "invalid_audio"),
])
def test_backend_failure_releases_slot_and_redacts_response_and_logs(tmp_path, caplog, error, status, category):
    async def run():
        backend = BlockingBackend(error)
        backend.release.set()
        app = create_app(backend=backend, service_role="worker", environ={}, temp_dir=tmp_path / "uploads")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
                response = await client.post("/v1/worker/transcribe",
                                             headers={"X-CtrlSpeak-Request-ID": "test_123"}, **upload())
                assert response.status_code == status
                assert response.headers["X-CtrlSpeak-Request-ID"] == "test_123"
                assert response.json()["detail"]["category"] == category
                assert response.json()["detail"]["request_id"] == "test_123"
                assert "private-" not in response.text
                assert app.state.admission.snapshot()["active"] == 0
                assert not backend.path.exists()
    with caplog.at_level(logging.INFO, logger="ctrlspeak_whisper_transcription"):
        asyncio.run(run())
    assert "request_id=test_123" in caplog.text
    assert "private-" not in caplog.text


def test_busy_probe_skips_upload_and_recovers_without_offline_circuit(tmp_path):
    clock = [10.0]
    calls = []
    busy = [True]
    def handle(request):
        calls.append(request.url.path)
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ready", "admission": {
                "available": 0 if busy[0] else 1, "status": "busy" if busy[0] else "ready"}})
        return httpx.Response(200, json={"raw_text": "ok"})
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = RemoteWorkerProvider("gpu", "http://worker", "never-expose", client=client,
                                        monotonic=lambda: clock[0])
        with pytest.raises(ProviderFailure) as caught:
            provider.transcribe(path, ProviderContext((), None, (), False, None))
        assert caught.value.category == "worker_busy"
        assert calls == ["/health"]
        assert provider.describe()["health"]["circuit_retry_after_ms"] == 0
        assert provider.describe()["status"] == "busy"
        clock[0] += 0.6
        busy[0] = False
        assert provider.transcribe(path, ProviderContext((), None, (), False, None))["raw_text"] == "ok"


@pytest.mark.parametrize("category,status", [("worker_busy", 503), ("worker_deadline_exceeded", 504)])
def test_busy_post_race_does_not_open_offline_circuit(tmp_path, category, status):
    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ready"})
        assert request.headers["X-CtrlSpeak-Request-ID"] == "request_123"
        assert "X-CtrlSpeak-OpenAI-Key" not in request.headers
        return httpx.Response(status, json={"detail": {"category": category}})
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = RemoteWorkerProvider("gpu", "http://worker", "worker", client=client)
        with pytest.raises(ProviderFailure) as caught:
            provider.transcribe(path, ProviderContext((), None, (), False, "secret", request_id="request_123"))
        assert caught.value.category == "worker_busy"
        assert provider.describe()["health"]["circuit_retry_after_ms"] == 0


@pytest.mark.parametrize("category,should_fallback", [("worker_busy", True), ("invalid_audio", False)])
def test_routing_falls_back_for_capacity_but_not_provider_independent_invalid_audio(tmp_path, category, should_fallback):
    calls = []
    class Provider:
        def __init__(self, name):
            self.id = name
        def transcribe(self, *_):
            calls.append(self.id)
            if self.id == "gpu":
                raise ProviderFailure("safe error", category=category,
                                      retryable=category == "worker_busy", status_code=422)
            return {"raw_text": "ok"}
    providers = [Provider("gpu"), Provider("cloud")]
    router = ProviderRouter(providers, {"preferred": ("gpu", "cloud")}, default_strategy="preferred")
    def transcribe():
        return router.transcribe(tmp_path / "unused.wav", ProviderContext((), None, (), False, None),
                                 strategy=None, provider_id=None, allowed_provider_ids={"gpu", "cloud"})
    if should_fallback:
        assert transcribe().provider_id == "cloud"
        assert calls == ["gpu", "cloud"]
    else:
        with pytest.raises(ProviderFailure, match="safe error"):
            transcribe()
        assert calls == ["gpu"]


def test_tiny_nonblocking_capacity_and_error_release(tmp_path):
    backend = BlockingBackend(RuntimeError("failed"))
    provider = LocalWhisperProvider("tiny", backend, lazy=True)
    lease = provider._admission.acquire()
    try:
        with pytest.raises(ProviderFailure) as caught:
            provider.transcribe(tmp_path / "unused", ProviderContext((), None, (), False, None))
        assert caught.value.category == "local_busy"
        assert provider.describe()["status"] == "busy"
    finally:
        lease.release()
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF")
    backend.release.set()
    with pytest.raises(ProviderFailure):
        provider.transcribe(path, ProviderContext((), None, (), False, None))
    assert provider._admission.snapshot()["active"] == 0


def test_worker_timings_and_correlation_are_safe_and_disk_pressure_is_visible(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Disk", (), {"total": 100, "free": 1})())
    async def run():
        backend = BlockingBackend()
        backend.release.set()
        app = create_app(backend=backend, service_role="worker", environ={}, temp_dir=tmp_path / "uploads")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
                response = await client.post("/v1/worker/transcribe",
                                             headers={"X-CtrlSpeak-Request-ID": "bad!request"}, **upload())
                payload = response.json()
                assert len(payload["request_id"]) == 32
                assert payload["request_id"] == response.headers["X-CtrlSpeak-Request-ID"]
                assert set(payload["timings"]) == {"receive_duration_ms", "upload_spool_duration_ms",
                    "queue_duration_ms", "inference_duration_ms", "cleanup_duration_ms", "server_duration_ms"}
                assert all(value >= 0 for value in payload["timings"].values())
                health = (await client.get("/health")).json()
                assert health["status"] == "ready"
                assert health["resources"]["disk_warning"] is True
                assert health["resources"]["disk_free_bytes"] == 1
    asyncio.run(run())
    assert safe_timings({"secret": "value", "queue_duration_ms": float("inf"),
                         "inference_duration_ms": 12, "cleanup_duration_ms": True}) == {"inference_duration_ms": 12.0}


def test_admission_release_is_idempotent_and_has_no_queue():
    admission = Admission(1, 60)
    lease = admission.acquire()
    assert admission.acquire() is None
    assert admission.snapshot()["queued"] == 0
    lease.release()
    lease.release()
    assert admission.snapshot()["active"] == 0


def test_forcibly_cancelled_task_still_cannot_release_live_thread(tmp_path):
    admission = Admission(1, 60)
    path = tmp_path / "active.wav"
    path.write_bytes(b"RIFF")
    owner = WorkOwner(admission.acquire(), path)
    entered, release = threading.Event(), threading.Event()
    def work():
        entered.set()
        assert release.wait(2)
        assert path.exists()
    thread = threading.Thread(target=lambda: owner.call(work), daemon=True)
    thread.start()
    try:
        assert entered.wait(1)
        owner.close()  # Same cleanup performed by a forcibly cancelled owner task.
        assert path.exists()
        assert admission.snapshot()["active"] == 1
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert not path.exists()
    assert admission.snapshot()["active"] == 0


def test_expired_gateway_budget_never_starts_another_provider(tmp_path):
    class Provider:
        id = "never"
        def transcribe(self, *_):
            pytest.fail("expired route must not start another inference")
    router = ProviderRouter([Provider()], {"route": ("never",)}, default_strategy="route")
    with pytest.raises(ProviderFailure) as caught:
        router.transcribe(tmp_path / "unused", ProviderContext((), None, (), False, None, deadline_at=0),
                          strategy=None, provider_id=None, allowed_provider_ids={"never"})
    assert caught.value.category == "request_deadline_exceeded"


@pytest.mark.parametrize("cancel_kind", ["disconnect", "task_cancel"])
def test_disconnected_gateway_never_starts_new_paid_fallback(tmp_path, cancel_kind):
    async def run():
        entered, release = threading.Event(), threading.Event()
        calls = []
        contexts = []
        class Provider:
            def __init__(self, name):
                self.id = name
            def transcribe(self, path, context):
                calls.append(self.id)
                contexts.append(context)
                if self.id == "gpu":
                    entered.set()
                    assert release.wait(3)
                    assert path.exists()
                    raise ProviderFailure("worker failed", category="worker_unavailable", retryable=True)
                pytest.fail("Disconnected request must not start paid cloud fallback")
        providers = [Provider("gpu"), Provider("paid-cloud")]
        app = create_app(backend=BlockingBackend(), service_role="gateway", environ={},
                         store=CorrectionStore(tmp_path / "gateway.sqlite3"),
                         router=ProviderRouter(providers, {"preferred": ("gpu", "paid-cloud")},
                                               default_strategy="preferred"),
                         temp_dir=tmp_path / "uploads")
        request = httpx.Request("POST", "http://gateway/v1/transcribe", **upload())
        body = request.read()
        messages = asyncio.Queue()
        messages.put_nowait({"type": "http.request", "body": body, "more_body": False})
        consumers = 0
        async def receive():
            nonlocal consumers
            consumers += 1
            assert consumers == 1, "disconnect monitoring must not race another body receiver"
            try:
                return await messages.get()
            finally:
                consumers -= 1
        async def send(_message):
            pass
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/v1/transcribe", "root_path": "",
                 "query_string": b"", "headers": [(key.lower(), value) for key, value in request.headers.raw],
                 "client": ("127.0.0.1", 1234), "server": ("gateway", 80)}
        async with app.router.lifespan_context(app):
            running = asyncio.create_task(app(scope, receive, send))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                if cancel_kind == "disconnect":
                    messages.put_nowait({"type": "http.disconnect"})
                    await asyncio.wait_for(running, 1)
                else:
                    running.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await running
                assert contexts[0].cancel_event.is_set()
                assert app.state.admission.snapshot()["active"] == 1
            finally:
                release.set()
                await wait_finished(app)
                if not running.done():
                    running.cancel()
                    await asyncio.gather(running, return_exceptions=True)
            assert calls == ["gpu"]
            assert app.state.admission.snapshot()["active"] == 0
            assert not list((tmp_path / "uploads").iterdir())
    asyncio.run(run())


def test_expired_spool_budget_never_schedules_worker_inference(tmp_path, monkeypatch):
    from starlette.datastructures import UploadFile
    original = UploadFile.read
    async def slow_read(self, *args):
        data = await original(self, *args)
        if data:
            await asyncio.sleep(0.12)
        return data
    monkeypatch.setattr(UploadFile, "read", slow_read)
    async def run():
        backend = BlockingBackend()
        backend.release.set()
        app = create_app(backend=backend, service_role="worker",
                         environ={"CTRLSPEAK_REQUEST_DEADLINE_SECONDS": "0.1"},
                         temp_dir=tmp_path / "uploads")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
                response = await client.post("/v1/worker/transcribe", **upload())
                assert response.status_code == 504
                assert not backend.entered.is_set()
                assert not app.state.owned_tasks
                assert app.state.admission.snapshot()["active"] == 0
                assert not list((tmp_path / "uploads").iterdir())
    asyncio.run(run())


def test_gateway_rejects_overlap_before_second_probe_or_wan_upload(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def handle(request):
        calls.append(request.url.path)
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ready"})
        entered.set()
        assert release.wait(3)
        return httpx.Response(200, json={"raw_text": "ok"})
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = RemoteWorkerProvider("gpu", "http://worker", "worker", client=client)
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(provider.transcribe, path, ProviderContext((), None, (), False, None))
            try:
                assert entered.wait(1)
                with pytest.raises(ProviderFailure) as caught:
                    provider.transcribe(path, ProviderContext((), None, (), False, None))
                assert caught.value.category == "worker_busy"
                assert provider.describe()["status"] == "busy"
                assert calls == ["/health", "/v1/worker/transcribe"]
            finally:
                release.set()
                assert first.result(timeout=3)["raw_text"] == "ok"
        assert provider.describe()["status"] == "ready"
