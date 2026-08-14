# CtrlSpeak transcription service v0.6.0

This directory contains both CtrlSpeak server roles:

- Nova runs `CTRLSPEAK_SERVICE_ROLE=gateway` and owns client authentication,
  routing, known words/corrections, feedback, and audit records.
- The local Ubuntu GPU host runs `CTRLSPEAK_SERVICE_ROLE=worker` and exposes
  raw `large-v3-turbo` CUDA/float16 inference to the gateway over WireGuard.
- `standalone` preserves the v0.5 combined local-model/API behavior.

Runtime state, virtual environments, and credentials remain outside Git.

## Install profiles

Python 3.11 and 3.12 are supported.

Gateway (CPU-only tiny fallback, no NVIDIA wheels):

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gateway.txt
python -m pip install -e . --no-deps
```

GPU worker:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-worker.txt
python -m pip install -e . --no-deps
```

Run tests with `python -m pytest -q`. `scripts/runtime-config` validates and
prints a redacted configuration summary.

## Worker configuration

Use a protected systemd drop-in. Bind only to the WireGuard/LAN interface
needed by the gateway when practical.

```ini
[Service]
Environment=CTRLSPEAK_SERVICE_ROLE=worker
Environment=CTRLSPEAK_WORKER_TOKEN=replace-with-a-dedicated-random-secret
Environment=WHISPER_BIND_HOST=10.83.233.2
Environment=WHISPER_PORT=8765
Environment=WHISPER_MODEL_NAME=large-v3-turbo
Environment=WHISPER_DEVICE=cuda
Environment=WHISPER_COMPUTE_TYPE=float16
```

The worker publishes `/health`, `/v1/capabilities`, and
`/v1/worker/transcribe`. It does not publish client transcription,
correction, or feedback routes.

## Gateway configuration

```ini
[Service]
Environment=CTRLSPEAK_SERVICE_ROLE=gateway
Environment=WHISPER_BIND_HOST=0.0.0.0
Environment=WHISPER_PORT=8765
Environment=CTRLSPEAK_CLIENTS_JSON={"wouter":{"token":"replace-client-token","admin":true,"providers":["*"]}}
Environment=CTRLSPEAK_WORKER_URL=http://10.83.233.2:8765
Environment=CTRLSPEAK_WORKER_TOKEN=replace-with-the-worker-secret
Environment=CTRLSPEAK_DEFAULT_STRATEGY=resilient-quality
Environment=CTRLSPEAK_FALLBACK_MODEL=tiny
Environment=CTRLSPEAK_FALLBACK_COMPUTE_TYPE=int8
Environment=CTRLSPEAK_FALLBACK_CPU_THREADS=2
```

Systemd quoting rules apply; for production, an `EnvironmentFile` with mode
`0600` is usually easier and safer for JSON and secrets. Never configure a
server-owned OpenAI key. The client supplies its own key on an individual
request and the gateway keeps it only for that call.

The default quality cascade is Ubuntu GPU → OpenAI `gpt-transcribe` with the
caller's key → local CPU `tiny`. Invalid keys and exhausted OpenAI quota are
terminal; retryable availability failures may fall through.

Install/reload the user unit:

```bash
scripts/install-user-service
systemctl --user daemon-reload
systemctl --user restart whisper-transcription.service
systemctl --user status --no-pager whisper-transcription.service
```

See the repository-level `docs/API.md` for the full request/response,
authentication, capability, routing, correction, and error contracts. Swagger
UI is available at `/docs`, ReDoc at `/redoc`, and OpenAPI at `/openapi.json`.

The Hermes command adapter now targets the gateway. Set
`CTRLSPEAK_GATEWAY_URL` and a dedicated `CTRLSPEAK_CLIENT_TOKEN`; do not point
it at the Ubuntu worker route.
