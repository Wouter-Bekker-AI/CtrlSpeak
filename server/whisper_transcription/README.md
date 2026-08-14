# CtrlSpeak Whisper Transcription API v0.5.3

This directory is the maintained Ubuntu backend for CtrlSpeak. It defaults to
`large-v3-turbo` with faster-whisper on CUDA float16. A deployment may instead
explicitly select CPU/int8; there is never an automatic device or compute-type
fallback. The API supports ordered, server-enforced output-language allowlists
as of v0.5.1.

The desktop client and this service share a release version, but they have
different roles: the Windows/Linux desktop records and inserts text; this
service performs GPU transcription and stores correction/audit records. Runtime
state (`data/`), the virtual environment, and service credentials are excluded
from Git.

## Python and installation

Supported service runtimes are Python 3.11 and 3.12. The primary production
target is Ubuntu 22.04 with a working NVIDIA driver and CUDA-capable GPU. The
OpenStack CPU deployment uses Ubuntu 24.04 and Python 3.12.

```bash
cd server/whisper_transcription
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

Run on loopback for local testing:

```bash
WHISPER_BIND_HOST=127.0.0.1 scripts/run-service
```

Run the redacted configuration summary with `scripts/runtime-config`. It never
prints the token value.

CUDA float16 is the default. To run an explicitly configured CPU service, set:

```ini
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_CPU_THREADS=4
WHISPER_NUM_WORKERS=1
```

`WHISPER_MODEL_NAME` defaults to `large-v3-turbo`. Startup fails closed when
the selected device or compute type is unavailable; the service never silently
changes from CUDA to CPU or vice versa.

## User systemd service and LAN access

`scripts/install-user-service` installs a user unit with loopback binding by
default. For an intended trusted-LAN deployment, create a protected drop-in:

```bash
systemctl --user edit whisper-transcription.service
```

```ini
[Service]
Environment=WHISPER_BIND_HOST=0.0.0.0
Environment=WHISPER_BEARER_TOKEN=replace-with-a-long-random-secret
```

Then run:

```bash
systemctl --user daemon-reload
systemctl --user enable --now whisper-transcription.service
systemctl --user status whisper-transcription.service
```

Non-loopback requests are rejected unless `WHISPER_BEARER_TOKEN` is configured
and the request supplies `Authorization: Bearer ...`. Loopback requests do not
require the bearer token. Use a trusted LAN/VPN or add an HTTPS reverse proxy;
plain HTTP does not protect audio or credentials on an untrusted network.

## Output-language contract

`POST /v1/transcribe` accepts `allowed_languages` as an ordered,
comma-separated list of one to five Whisper language codes.

- Omitted or empty: automatic Whisper language selection, with no restriction.
- One code, such as `en`: the server forces that decoding language.
- Several codes, such as `en,af`: the server detects the spoken language once;
  if it is in the list, that code is forced, otherwise the first code is the
  fallback.
- A response whose reported language is outside a non-empty allowlist is
  blocked by the server.

The older single `language` multipart field remains supported. If both fields
are supplied, `language` must occur in `allowed_languages`. The language
setting controls transcription/recognition; it is not arbitrary translation
between languages.

See the repository-level `docs/API.md` for every route, fields, examples, and
response/error contracts. Interactive OpenAPI documentation is available at
`/docs`, the ReDoc view at `/redoc`, and the OpenAPI document at
`/openapi.json` while the service is running.

## Runtime data and limits

By default the service stores model files, uploads, SQLite corrections, and
transcription audit records below `data/`. Override this with
`WHISPER_DATA_DIR`. Temporary uploads are deleted after each request. The
default upload limit is 100 MiB and can be changed with
`WHISPER_MAX_UPLOAD_BYTES`.

The health endpoint reports only readiness, service version, model, device, and
compute type. It does not expose credentials or transcription content.

## Hermes adapter compatibility

`scripts/hermes-stt-api INPUT_AUDIO LANGUAGE OUTPUT_TEXT` preserves the existing
Hermes command-provider integration. It reads the user-service LAN drop-in when
present, sends the requested single language through the
`allowed_languages` contract, and writes only the corrected transcript to the
requested output file. Its established downstream name-normalization rules are
retained. The adapter never prints the bearer value.
