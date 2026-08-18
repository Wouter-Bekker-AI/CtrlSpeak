# CtrlSpeak API v0.7.0

The maintained implementation is in `server/whisper_transcription`. The v0.7
contract preserves the three deployment roles introduced in v0.6 and adds safe,
measured routing telemetry for Midnight Signal. The roles are:

- `gateway`: client-facing orchestration, identities, provider routing,
  language enforcement, corrections, feedback, and audit records;
- `worker`: service-to-service raw inference only; no client transcription
  route, correction routes, or audit storage; and
- `standalone`: backward-compatible local model plus gateway API for
  development or a single trusted server.

The intended production topology is the dedicated OpenStack `CtrlSpeak`
instance in `gateway` mode. Its default `ubuntu-gpu-preferred` strategy orders
`ubuntu-gpu-large-v3-turbo` → `openai-gpt-transcribe` →
`nova-tiny-whisper`; `openai-preferred` reverses the first two providers. The
last identifier is retained for API compatibility, but that CPU fallback runs
on `CtrlSpeak`, not Nova. The Ubuntu GPU machine is a `worker`; Nova is only
the WireGuard transport hub.

## Authentication and identity

Non-loopback gateway requests use a client bearer identity. Configure a
protected JSON object in `CTRLSPEAK_CLIENTS_JSON`:

```json
{
  "alice": {
    "token": "long-random-client-token",
    "admin": false,
    "providers": ["*"]
  }
}
```

The legacy `WHISPER_BEARER_TOKEN` remains supported as an administrator named
`legacy`. Each token maps to a principal ID; user corrections, exact overrides,
transcriptions, and feedback are isolated by that identity. Tokens are never
returned by health, capabilities, logs, or audit metadata.

The worker requires a separate `CTRLSPEAK_WORKER_TOKEN`. The dedicated
CtrlSpeak gateway uses that token when calling the worker. A client token
cannot call the worker, and the worker token is not a client identity.

Use HTTPS or a trusted private VPN. The production
CtrlSpeak-gateway-to-Ubuntu path uses WireGuard through Nova; bearer tokens do
not encrypt HTTP by themselves.

## Health and capabilities

`GET /health` returns the service version and role. A worker also reports its
loaded model/runtime:

```json
{
  "status": "ready",
  "version": "0.7.0",
  "role": "worker",
  "model": "large-v3-turbo",
  "device": "cuda",
  "compute_type": "float16"
}
```

`GET /v1/capabilities` is authenticated. A gateway response contains only the
providers permitted for the caller:

```json
{
  "version": "0.7.0",
  "role": "gateway",
  "accepts_client_transcriptions": true,
  "applies_corrections": true,
  "openai_key_storage": "request_only",
  "telemetry": {
    "attempt_duration_ms": true,
    "routing_duration_ms": true,
    "worker_inference_duration_ms": true,
    "worker_health": true
  },
  "default_strategy": "ubuntu-gpu-preferred",
  "providers": [
    {
      "id": "ubuntu-gpu-large-v3-turbo",
      "kind": "ctrlspeak_worker",
      "model": "large-v3-turbo",
      "device": "cuda",
      "status": "ready",
      "requires_credential": false,
      "paid": false,
      "fast_failover": true,
      "health_cache_seconds": 5.0,
      "circuit_break_seconds": 30.0,
      "health": {
        "status": "ready",
        "probe_status": "ready",
        "probe_duration_ms": 18.4,
        "probe_age_ms": 121.7,
        "probe_timeout_ms": 500.0,
        "connect_timeout_ms": 350.0,
        "cache_remaining_ms": 4878.3,
        "circuit_retry_after_ms": 0.0
      }
    },
    {
      "id": "openai-gpt-transcribe",
      "kind": "openai",
      "model": "gpt-transcribe",
      "status": "available_with_key",
      "requires_credential": true,
      "credential_header": "X-CtrlSpeak-OpenAI-Key",
      "paid": true
    },
    {
      "id": "nova-tiny-whisper",
      "kind": "local_whisper",
      "model": "tiny",
      "device": "cpu",
      "status": "available",
      "requires_credential": false,
      "paid": false
    }
  ],
  "strategies": [
    {
      "id": "ubuntu-gpu-preferred",
      "providers": [
        "ubuntu-gpu-large-v3-turbo",
        "openai-gpt-transcribe",
        "nova-tiny-whisper"
      ]
    },
    {
      "id": "openai-preferred",
      "providers": [
        "openai-gpt-transcribe",
        "ubuntu-gpu-large-v3-turbo",
        "nova-tiny-whisper"
      ]
    },
    {
      "id": "ubuntu-gpu-only",
      "providers": ["ubuntu-gpu-large-v3-turbo"]
    },
    {
      "id": "openai-only",
      "providers": ["openai-gpt-transcribe"]
    },
    {
      "id": "gateway-tiny-only",
      "providers": ["nova-tiny-whisper"]
    }
  ]
}
```

A worker capability response sets `accepts_client_transcriptions` and
`applies_corrections` to `false`. The desktop uses this to prevent accidental
configuration of a worker as its public backend.

## Client transcription

`POST /v1/transcribe` exists on gateway and standalone roles only and uses
`multipart/form-data`.

| Field | Required | Meaning |
| --- | --- | --- |
| `audio` | yes | Supported audio file; the desktop sends WAV. |
| `allowed_languages` | no | Ordered comma-separated language codes; maximum five. |
| `language` | no | Legacy single-code alias. |
| `initial_prompt` | no | Optional caller vocabulary/context. |
| `context` | no | Optional deterministic correction context. |
| `word_timestamps` | no | Request timestamps from providers that support them. |
| `strategy` | no | A strategy ID returned by capabilities. |
| `provider` | no | One explicit provider ID; mutually exclusive with `strategy`. |

If neither route field is supplied, the gateway uses its default strategy.
Clients cannot submit arbitrary upstream URLs.

To use `openai-gpt-transcribe`, send the caller's key on that request only:

```http
X-CtrlSpeak-OpenAI-Key: <caller's OpenAI API key>
```

The Windows desktop can keep this key in the current user's Windows Credential
Manager vault and loads it into process memory on startup. It does not save the
key to `settings.json` or send it to the Ubuntu worker. The gateway does not
persist it or include it in logs/audit metadata. This isolates clients from
one another's keys; as with any proxy, the gateway administrator controls the
process handling the request and must be trusted. Platforms without a
supported native credential vault remain session-only.

Example:

```bash
curl -H "Authorization: Bearer $CTRLSPEAK_CLIENT_TOKEN" \
  -H "X-CtrlSpeak-OpenAI-Key: $OPENAI_API_KEY" \
  -F "audio=@sample.wav" \
  -F "allowed_languages=en,af" \
  -F "strategy=ubuntu-gpu-preferred" \
  https://ctrlspeak.example/v1/transcribe
```

Successful response fields are additive to v0.5:

```json
{
  "id": "transcription-id",
  "raw_text": "raw model text",
  "text": "text after gateway corrections",
  "language": "en",
  "detected_language": "en",
  "detected_languages": ["en"],
  "allowed_languages": ["en", "af"],
  "language_policy": "restricted",
  "segments": [],
  "applied_correction_rule_ids": [],
  "exact_override_id": null,
  "provider_used": "openai-gpt-transcribe",
  "requested_strategy": "ubuntu-gpu-preferred",
  "attempts": [
    {
      "provider": "ubuntu-gpu-large-v3-turbo",
      "status": "failed",
      "category": "worker_unavailable",
      "retryable": true,
      "duration_ms": 351.2
    },
    {
      "provider": "openai-gpt-transcribe",
      "status": "succeeded",
      "duration_ms": 842.6
    }
  ],
  "degraded": true,
  "routing_duration_ms": 1194.1,
  "usage": {"type": "tokens", "total_tokens": 120}
}
```

### Telemetry semantics

All `*_ms` values are non-negative milliseconds measured by the component that
performed the work. They are optional/additive protocol fields:

- `health.probe_duration_ms` is the gateway-observed Ubuntu worker health
  request duration; `probe_age_ms`, `cache_remaining_ms`, and
  `circuit_retry_after_ms` describe the current bounded fast-failover state.
- `attempts[].duration_ms` is time spent by the gateway on that provider
  attempt, including its provider-specific network and validation work.
- `attempts[].inference_duration_ms`, when present for the Ubuntu worker, is
  measured inside the worker around model inference. It is not network or total
  request latency.
- `routing_duration_ms` is gateway time from beginning the selected route until
  success or exhaustion. It can be longer than inference and is not presented
  as inference by the desktop.

An omitted/null value means the measurement is unavailable. Clients must show
that honestly rather than substituting a sample value or relabelling their own
wall-clock timer. Capability flags state which telemetry families this server
supports. These structures contain provider IDs, bounded public status/category
values, and timings only; they must not contain audio, transcript text, tokens,
OpenAI keys, prompts, correction phrases, upstream response bodies, or internal
exception messages.

The five primary strategies are:

- `ubuntu-gpu-preferred`: Ubuntu GPU → OpenAI → gateway tiny;
- `openai-preferred`: OpenAI → Ubuntu GPU → gateway tiny;
- `ubuntu-gpu-only`: Ubuntu GPU with no fallback;
- `openai-only`: OpenAI with no fallback; and
- `gateway-tiny-only`: gateway CPU tiny with no upstream provider.

Explicit cascading strategies continue after both retryable availability
failures and terminal upstream conditions such as an invalid/missing OpenAI key
or exhausted credit. Each failure remains in `attempts`. Single-provider routes
preserve the provider's exact actionable `401`, `402`, `429`, or availability
error. Compatibility aliases from v0.6.0/v0.6.1 remain published.

Before uploading audio to the Ubuntu worker, the gateway performs a fast health
probe. The default health deadline is 500 ms with a 350-ms connection ceiling.
Failure opens a 30-second circuit; requests during that window record
`worker_circuit_open` and immediately continue to the next provider. A healthy
probe is cached for five seconds. Operators can tune these bounded values with
`CTRLSPEAK_WORKER_HEALTH_TIMEOUT_SECONDS`,
`CTRLSPEAK_WORKER_CONNECT_TIMEOUT_SECONDS`,
`CTRLSPEAK_WORKER_HEALTH_CACHE_SECONDS`, and
`CTRLSPEAK_WORKER_CIRCUIT_BREAK_SECONDS`.

## Language enforcement

One allowed code forces that language. Several codes constrain detection to
that ordered set, with the first as the safe local fallback. `gpt-transcribe`
receives the same set through its supported `languages[]` field. The gateway
blocks provider metadata outside a non-empty allowlist. This controls
recognition/transcription language; it is not translation.

## Known words and corrections

Gateway correction routes are:

- `POST /v1/corrections`
- `GET /v1/corrections?enabled=true|false`
- `GET /v1/corrections/{rule_id}`
- `PATCH /v1/corrections/{rule_id}`
- `DELETE /v1/corrections/{rule_id}`

Create example:

```json
{
  "source_phrase": "Acme corp",
  "replacement_phrase": "ACME Corp",
  "context_terms": [],
  "tags": ["company"],
  "enabled": true,
  "priority": 0,
  "scope": "user",
  "language_codes": ["en"],
  "send_as_keyword": true
}
```

`scope` is `user` by default. Only an administrator may create `global` rules.
When `send_as_keyword` is enabled, applicable source and replacement phrases
are sent as bounded provider hints. The gateway still performs deterministic,
single-pass replacement after raw transcription because hints are not a
guarantee. The worker never applies corrections.

## Feedback

`POST /v1/transcriptions/{id}/feedback` attaches rule feedback and/or approves
an exact raw-transcript override for the authenticated principal. The desktop
binds feedback to the original transcription ID, URL, and client bearer token.

```json
{
  "rule_ids": [],
  "confirmed_text": "the user-confirmed final text",
  "capture_method": "active_field_on_enter",
  "client_metadata": {"client": "CtrlSpeak", "version": "0.7.0"}
}
```

## Worker transcription

`POST /v1/worker/transcribe` exists only in worker role and requires the worker
bearer token. Fields are `audio`, `allowed_languages`, legacy `language`,
`initial_prompt`, JSON-string-array `keywords`, and `word_timestamps`. It
returns raw model output plus `provider`, measured `inference_duration_ms`, and
`corrected: false`. It never stores the audio, transcript, client identity,
correction, or OpenAI credential.

## Errors and interactive documentation

Invalid fields return `422`; oversized audio returns `413`; unauthenticated or
unauthorised requests return `401`/`403`; upstream policy violations return
`502`; and exhausted retryable providers return `503`. Provider error details
contain safe category, retryability, attempts, and optional action fields, not
secrets or upstream response bodies.

The running service publishes OpenAPI at `/openapi.json`, Swagger UI at
`/docs`, and ReDoc at `/redoc`.
