# CtrlSpeak transcription service

This directory contains the maintained CtrlSpeak service and command helper:

- The dedicated OpenStack `CtrlSpeak` instance runs
  `CTRLSPEAK_SERVICE_ROLE=gateway` and owns client authentication, routing,
  known words/corrections, feedback, and audit records.
- The local Ubuntu GPU host runs `CTRLSPEAK_SERVICE_ROLE=worker` and exposes
  raw `large-v3-turbo` CUDA/float16 inference to the gateway over WireGuard.
- Nova runs the external STT helper in an isolated environment. It calls the
  gateway and is not a CtrlSpeak server. The deployment leaves Hermes itself
  untouched.
- `standalone` preserves the v0.5 combined local-model/API behavior.

Runtime state, virtual environments and credentials remain outside Git. The
[signed release workflow](../../docs/UPDATING.md#maintainer-release-workflow)
stages immutable Git checkouts and runs health checks and canaries before
publication. The desktop installer does not deploy services. For the wire
contract, including WAV compatibility, FLAC pass-through, bounded admission
and S1 metadata, see the [API guide](../../docs/API.md).

## Worker-side S1 cleanup

Only a CUDA worker can enable S1. Requests must send `cleanup=true`; omission
and false skip it. The gateway validates the optional cleaned result and applies
its correction rules. OpenAI/Tiny return no cleaned variant, and stale gateway
CPU-normalization settings cannot enable cleanup there.

Stage verified assets with repository-root `scripts/stage_s1_gpu.py` on Ubuntu.
Build `docker build -f Dockerfile.s1-runtime -t ctrlspeak-s1-runtime:0.7.4 .`
with the host's existing NVIDIA Container Toolkit, then install the user unit
`systemd/ctrlspeak-s1-mini.service`. The pinned Ubuntu 24.04 container supplies
the upstream binary's required glibc/libstdc++ without changing Ubuntu 22.04
host libraries. Model/runtime mounts are read-only; only host loopback 8081 is
published. Adjust the unit's numeric user/group if not 1000. Model/runtime stay
outside source deployments. Verify `--list-devices` reports the intended GPU.
Worker-only environment additions:

```ini
Environment=CTRLSPEAK_NORMALIZATION_ENABLED=true
Environment=CTRLSPEAK_NORMALIZATION_URL=http://127.0.0.1:8081/v1/chat/completions
Environment=CTRLSPEAK_NORMALIZATION_TIMEOUT_SECONDS=3
```

Verify actual CUDA layer offload before enabling cleanup. Keep S1 warm; its
failure must not stop Whisper. After verifying replacement, disable the old
gateway S1 CPU service and its health timer; retain source/configuration/model
rollback copies. See the repository API and 0.7.4 release record for acceptance.

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

In a test environment install `pytest==8.4.2`, then run
`python ../../scripts/run_bounded_tests.py --timeout 180 --cwd . -q` from
this directory. The full suite includes Linux shell-runtime tests.
`scripts/runtime-config` validates and prints a redacted configuration summary.

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
Environment=WHISPER_BIND_HOST=<private-gateway-address>
Environment=WHISPER_PORT=8765
Environment=CTRLSPEAK_CLIENTS_JSON={"wouter":{"token":"replace-client-token","admin":true,"providers":["*"]}}
Environment=CTRLSPEAK_WORKER_URL=http://10.83.233.2:8765
Environment=CTRLSPEAK_WORKER_TOKEN=replace-with-the-worker-secret
Environment=CTRLSPEAK_DEFAULT_STRATEGY=ubuntu-gpu-preferred
Environment=CTRLSPEAK_WORKER_CONNECT_TIMEOUT_SECONDS=0.35
Environment=CTRLSPEAK_WORKER_HEALTH_TIMEOUT_SECONDS=0.5
Environment=CTRLSPEAK_WORKER_HEALTH_CACHE_SECONDS=5
Environment=CTRLSPEAK_WORKER_CIRCUIT_BREAK_SECONDS=30
Environment=CTRLSPEAK_FALLBACK_MODEL=tiny
Environment=CTRLSPEAK_FALLBACK_COMPUTE_TYPE=int8
Environment=CTRLSPEAK_FALLBACK_CPU_THREADS=2
```

Systemd quoting rules apply; for production, an `EnvironmentFile` with mode
`0600` is usually easier and safer for JSON and secrets. Never configure a
server-owned OpenAI key. The client supplies its own key on an individual
request and the gateway keeps it only for that call.

The default quality cascade is Ubuntu GPU → OpenAI `gpt-transcribe` with the
caller's key → local CPU `tiny`. `openai-preferred` reverses the first two;
`ubuntu-gpu-only`, `openai-only`, and `gateway-tiny-only` disable fallback.
Cascading routes continue after unavailable providers, invalid/missing OpenAI
keys, and exhausted quota while preserving those failures in response
metadata. Single-provider routes return the exact provider error.

The worker health probe has a 500-ms total deadline and 350-ms connection
ceiling by default. A failed probe opens a 30-second circuit so later requests
fail over immediately; a healthy result is cached for five seconds. The long
provider timeout applies only after a worker connection has been established
for actual inference.

`GET /v1/capabilities` exposes safe provider-health telemetry for the worker,
including its latest bounded probe status and duration, cache lifetime, and
circuit retry delay. Every `/v1/transcribe` attempt includes a measured
`duration_ms`; worker successes also include the worker-observed
`inference_duration_ms`. The response and stored audit metadata include the
complete `routing_duration_ms`. These values are server-observed timings and
never include audio, transcript text, API keys, or bearer credentials.

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

The external Hermes command adapter targets the gateway. Install
`requirements-helper.txt` in the isolated helper environment and run
`scripts/hermes-stt-api --self-test` there before selecting a release.
Set `CTRLSPEAK_GATEWAY_URL` and a dedicated `CTRLSPEAK_CLIENT_TOKEN` in its
protected configuration; do not point it at the Ubuntu worker route or install
its dependencies into Hermes.
