# CtrlSpeak API v0.8.4

0.8.4 changes desktop microphone-level presentation and Windows shortcut tracking.
The saved `microphone_level_dbfs` preference is desktop-only and is not sent to
the gateway. All transcription/correction/cleanup/authentication and external
helper contracts are unchanged; fleet version metadata reports 0.8.4. See
[0.8.4 release details](V0.8.4_RELEASE.md).

0.8.3 adds a simulation-only desktop keyboard guide. Gateway, worker and external
helper report 0.8.3; no transcription endpoint, authentication, correction,
cleanup or Hermes integration contract changed. The guide does not call the API.
See [0.8.3 guide and patch-update acceptance](V0.8.3_RELEASE.md).

0.8.2 introduces desktop differential updates, not a transcription API change.
Gateway, worker and external-helper version metadata is aligned; requests,
authentication, language constraints, correction rules, cleanup and response
contracts are unchanged. Older clients remain compatible. See
[0.8.2 updater/release details](V0.8.2_RELEASE.md).

Previous patch 0.8.1 changed desktop Ctrl-key recognition and Live overlay placement.
The batch transcription, correction, language, cleanup and helper contracts are
unchanged. Its fleet version was aligned at 0.8.1; 0.7.7/0.8.0 clients remain
compatible. See [0.8.1 release details](V0.8.1_RELEASE.md).

## Desktop dictation modes and compatibility (0.8.0)

The new desktop modes do **not** introduce a streaming API, session socket,
new authentication scheme or new transcription response schema. Hold to Talk
and Hands-free submit a completed recording through the existing endpoint.
Opt-in Windows Live Dictation segments speech locally after approximately
650 ms silence, or at a maximum eight-second phrase, then submits one ordinary
transcription request per phrase. Network and inference time are additional.

Each phrase uses the existing lossless audio transport, caller identity,
provider/strategy, caller-owned OpenAI key, language policy, correction rules
and optional `cleanup` request. Paid-provider usage and cleanup overhead can
occur per phrase. Requests are serialized by the desktop session with a bounded
waiting queue; this does not remove server admission limits or imply zero
latency. Cancellation cannot recall audio already accepted by a provider or
guarantee that it will not be billed.

Live's single-line formatting and trusted-space separator are **desktop-only**
insertion policy, applied after normal validated selection of cleaned/ordinary
text. The gateway does not flatten `text`, `normalized_text` or
`s1_cleaned_text`, and the worker does not change S1 formatting. Hold/Hands-free
continue using the saved desktop formatting preference. The client-side
`live_chunk` metadata flag is not a new public multipart request field.

Live phrases are not registered for automatic whole-field Enter feedback:
a later edited field cannot safely be attributed to just the last phrase.
Existing explicit correction and feedback API endpoints remain available.

Nova/Hermes continues calling the external CtrlSpeak command helper for each
complete voice note. It should keep selecting one validated cleaned transcript
when applied, otherwise ordinary `text`; no new mode parameters or transcript
echo logic are needed. The coordinated release updates gateway, Ubuntu worker
and the external helper at the same signed revision without patching or
restarting Hermes. Existing 0.7.7 clients remain API-compatible, including a
desktop deliberately rolled back to 0.7.7. See the
[0.8.0 release and operations guide](V0.8.0_RELEASE.md).

## Admission, deadlines and diagnostics (0.7.7)

The gateway/worker retain the existing transcription and correction contracts.
New request correlation and numeric stage timing fields are additive; consumers
must continue selecting `normalized_text` only with validated applied S1
provenance, otherwise `text`. No desktop formatting policy is imposed on API
or Telegram output.

Workers have bounded inference capacity and explicitly report busy/stalled
admission state. The gateway treats this as a retryable provider-availability
condition and follows the caller's configured strategy; GPU-only requests do
not silently use a paid alternative. Provider-independent invalid audio stops
the chain instead of causing redundant requests/charges.

Request response deadlines do not claim to forcibly terminate native GPU work.
An occupied slot and its audio remain owned until the actual operation finishes,
even when the waiting caller times out/disconnects. Subsequent requests can
bypass that busy worker. See [0.7.7 operations and exact limits](V0.7.7_RELEASE.md).

| Limit | Default | Override |
| --- | --- | --- |
| Worker / gateway active requests | 1 / 8 | `CTRLSPEAK_MAX_ACTIVE_REQUESTS` (1–32) |
| Worker / gateway response deadline | 75 / 150 seconds | `CTRLSPEAK_REQUEST_DEADLINE_SECONDS` (greater than 0, at most 600) |
| Gateway Tiny inference capacity | 1 | Fixed |
| Busy / offline provider cooldown | 0.5 / 30 seconds | Busy is distinct from offline |

Send `X-CtrlSpeak-Request-ID` with 1–64 ASCII letters, digits, `_` or `-` to
correlate a request. Invalid/missing IDs are replaced. The response header and
JSON `request_id` identify the request. Transcription responses add numeric
`timings`: `receive_duration_ms`, `upload_spool_duration_ms`,
`routing_duration_ms`, `server_duration_ms`, and (when using the worker) nested
`worker` measurements including `queue_duration_ms`, `inference_duration_ms`
and `cleanup_duration_ms`. Actual multipart receive and post-parse upload spool
are distinct. Inference timing includes decode/thread dispatch, not just GPU
kernel execution. Do not add these overlapping measurements together.

Busy/deadline failures provide typed categories such as `worker_busy`,
`gateway_busy`, `local_busy`, `worker_deadline_exceeded` and
`request_deadline_exceeded`; retryable does not mean automatically replay.
An earlier provider may have accepted the audio before a timeout.
Health `admission` reports capacity/active/available/oldest-active age/status.
`resources.disk_warning` is true below 5 GiB free or above 90% used; this is a
diagnostic warning, not automatic storage cleanup.

## Formatted text and audit (0.7.6)

The gateway preserves formatted API text; desktop injection safety is an edge
preference, not a gateway transformation. New additive response field
`s1_cleaned_text` contains validated S1 output before dictionary rules (or null).
`normalized_text` remains the final optional cleaned variant, with paragraphs
intact. Consumers must continue selecting it only with the existing provenance
checks. `s1_cleaned_text` is diagnostic, not a new selection field: an exact
override can suppress `normalized_text` while retaining S1 evidence.

Both values are now retained with raw/corrected text in the protected gateway
audit database. Historical rows have null new fields. Authorization and API
routes are unchanged; no public transcript-history endpoint is introduced.
Do not include transcript contents in ordinary operational log exports.

Windows/Linux desktop clients default to single-line safe insertion. Their
optional **Preserve paragraphs and line breaks** checkbox is only shown with
S1 cleanup enabled, and only takes effect for actually applied S1 results.
Nova's external helper and Telegram formatting remain unchanged. See
[0.7.6 release/privacy guide](V0.7.6_RELEASE.md).

## Lossless transport (0.7.5)

The multipart `audio` field accepts WAV and FLAC, plus existing Ogg/Opus, MP3,
M4A and WebM containers supported by the inference decoder. Send a matching
filename extension and MIME type (`recording.flac`, `audio/flac`, or
`recording.ogg`, `audio/ogg`). Do not put a compression layer around the whole
multipart body. Existing WAV callers remain compatible.

CtrlSpeak senders encode PCM16 mono/stereo WAV to FLAC preserving sample rate,
channel count, sample count and every integer sample. Already-compressed input
is never transcoded. The gateway forwards the original compressed file on both
the GPU-worker and OpenAI routes, including fallback attempts. Only the selected
local inference backend decodes/resamples for Whisper. OpenAI accepts FLAC/Ogg
directly ([official reference](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)).

`/health` and `/v1/capabilities` add `revision` (deployed Git commit, or `source`)
and `audio_transport`: protocol `1`, lossless encoding `flac`, accepted formats,
compressed pass-through, upload limit 100 MiB, decoded PCM16-equivalent limit
100 MiB and maximum duration 1,800 seconds. Local decoders enforce actual sample
limits and a cooperative 20-second decode deadline. Invalid or excessive local
audio returns HTTP 422 and is not retried against a paid provider. OpenAI's own
upload/format limits also apply; these capability fields do not raise them.

The desktop records preparation time, original/upload byte counts and outcome
in diagnostic metadata, without audio content, file paths or credentials.
No extra capability request is added to each transcription. 0.7.4 inference
endpoints already support FLAC via PyAV, allowing a compatible rolling update.

The external Nova command helper is versioned with CtrlSpeak, not Hermes.
It retains one selected transcript and the existing schema-2 `.ctrlspeak.json`
feedback sidecar. See [Nova integration](NOVA_CTRLSPEAK_0.7.5_HANDOFF.md).

The maintained implementation is in `server/whisper_transcription`. The v0.7
contract preserves the three deployment roles introduced in v0.6 and adds safe,
measured routing telemetry for Midnight Signal. The roles are:

- `gateway`: client-facing orchestration, identities, provider routing,
  language enforcement, corrections, feedback, and audit records;
- `worker`: service-to-service raw inference only; no client transcription
  route, correction routes, or audit storage; and
- `standalone`: backward-compatible local model plus gateway API for
  development or a single trusted server.

v0.7.3 synchronizes the service identity with the desktop correction-dialog,
overlay-brand, and Windows clipboard hotfix. It does not change routes,
schemas, authentication, provider routing, corrections, language enforcement,
telemetry, or stored data.

The intended production topology is the dedicated OpenStack `CtrlSpeak`
instance in `gateway` mode. Its default `ubuntu-gpu-preferred` strategy orders
`ubuntu-gpu-large-v3-turbo` → `openai-gpt-transcribe` →
`nova-tiny-whisper`; `openai-preferred` reverses the first two providers. The
last identifier is retained for API compatibility, but that CPU fallback runs
on `CtrlSpeak`, not Nova. The Ubuntu GPU machine is a `worker`; Nova is only
the WireGuard transport hub.

## Optional GPU cleanup (0.7.4)

Both transcription endpoints accept multipart `cleanup` (boolean, default
`false`). Only `ubuntu-gpu-large-v3-turbo` can perform cleanup, on the Ubuntu
CUDA worker after Whisper in the same request. OpenAI, gateway Tiny and
standalone inference never invoke S1. The gateway never runs a normalizer.

Gateway capabilities include `text_normalization.mode=optional-worker-gpu`,
`request_field=cleanup`, `default=false`, `eligible_providers`, `device=cuda`,
`location=ubuntu-worker`, `language=en`, and `gateway_cpu_cleanup=false`. Worker
capabilities state whether its normalizer is configured; that does not promise
cleanup availability for every request.

Response additions (illustrative timing):

```json
{
  "raw_text": "hello acme",
  "text": "hello ACME",
  "normalized_text": "Hello ACME.",
  "s1_cleaned_text": "Hello acme.",
  "normalized_applied_correction_rule_ids": ["rule-id"],
  "normalization": {
    "requested": true, "enabled": true, "applied": true, "status": "applied",
    "model": "S1-mini by Superwhisper", "language": "en",
    "location": "ubuntu-worker", "device": "cuda", "fail_open": true,
    "duration_ms": 180.0
  }
}
```

`enabled` records request opt-in. `applied` records the actual outcome.
`normalized_text` is null when cleanup was not applied; `text` remains the
ordinary deterministic-correction variant. Rules also apply to cleaned text
without double-counting usage. Exact user-confirmed overrides take precedence.

Select normalized text only when the caller requested cleanup, provider is
`ubuntu-gpu-large-v3-turbo`, metadata has `requested=true`, `applied=true`,
`status=applied`, `device=cuda`, `location=ubuntu-worker`, the expected model
attribution, and non-empty normalized text. Otherwise use `text`. Keep
`raw_text` and the transcription ID for corrections/feedback; never concatenate
both variants into an agent instruction.

Statuses are `disabled`, `applied`, `unchanged`, `unavailable`, `non_english`,
`empty_input`, `input_too_long`, `timeout`, `runtime_error`, `unsafe_output`,
and gateway `exact_override`. Unknown/malformed metadata falls back to `text`.
Cleanup failures never retry transcription or turn successful ASR into an error.
Default bounds: three seconds total, 4,000 input characters, bounded output,
no waiting queue and a 15-second failure cooldown. Timings contain no transcript.

Hermes/Nova: send `strategy=ubuntu-gpu-preferred` and `cleanup=true`; do not also
send `provider`. Supply the caller's OpenAI key so fallback works when Ubuntu
is offline. Display two transcript echoes only when cleanup actually changed
the selected text; otherwise display one.

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
  "version": "0.8.4",
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
  "version": "0.8.4",
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
  "client_metadata": {"client": "CtrlSpeak", "version": "0.8.4"}
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
