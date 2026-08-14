# CtrlSpeak Whisper API v0.5.3

The maintained API implementation is in `server/whisper_transcription`. Its
OpenAPI title and version are **CtrlSpeak Whisper Transcription API 0.5.3**.
All examples below use the default base URL `http://127.0.0.1:8765`; use the
URL configured for your deployment.

## Authentication

Loopback requests are allowed without authentication. Every non-loopback
request requires the service to have `WHISPER_BEARER_TOKEN` configured and the
client to send:

```http
Authorization: Bearer <token>
```

Missing configuration returns `403`; missing or invalid credentials return
`401`. Use HTTPS or a trusted VPN outside a trusted local network.

## Health

`GET /health` returns `503` until the explicitly configured model/runtime is
ready, otherwise:

```json
{
  "status": "ready",
  "version": "0.5.3",
  "model": "large-v3-turbo",
  "device": "cuda",
  "compute_type": "float16"
}
```

CUDA/float16 is the default. A CPU deployment explicitly configures
`WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8`; there is no automatic
fallback between runtimes.

## Transcribe audio

`POST /v1/transcribe` uses `multipart/form-data`.

| Field | Required | Meaning |
| --- | --- | --- |
| `audio` | yes | Audio file accepted by faster-whisper. CtrlSpeak sends WAV. |
| `allowed_languages` | no | Ordered comma-separated Whisper codes; maximum five. |
| `language` | no | Legacy single-code alias for `allowed_languages`. |
| `initial_prompt` | no | Optional Whisper vocabulary/context hint. |
| `context` | no | Optional correction-rule context. |
| `word_timestamps` | no | Boolean; include per-word timing when true. |

Language-policy examples:

```bash
# Automatic/unrestricted
curl -F "audio=@sample.wav" http://127.0.0.1:8765/v1/transcribe

# Force English
curl -F "audio=@sample.wav" -F "allowed_languages=en" \
  http://127.0.0.1:8765/v1/transcribe

# English or Afrikaans; English is the fallback
curl -F "audio=@sample.wav" -F "allowed_languages=en,af" \
  http://127.0.0.1:8765/v1/transcribe
```

For a single code, that decoding language is forced. For multiple codes, the
service accepts detected language only when it is allowed and otherwise forces
the first code. A final language outside the list is never returned. Omission
keeps automatic detection. This constrains recognition language; it does not
translate arbitrary speech into a requested target language.

Successful response:

```json
{
  "id": "transcription-id",
  "raw_text": "raw model text",
  "text": "text after approved corrections",
  "language": "en",
  "detected_language": "en",
  "allowed_languages": ["en", "af"],
  "language_policy": "restricted",
  "segments": [],
  "applied_correction_rule_ids": [],
  "exact_override_id": null
}
```

Invalid or conflicting language codes return `422`. Oversized audio returns
`413`. Model/transcription or policy-enforcement failures return `500` without
returning out-of-policy text. A not-yet-ready model returns `503`.

## Corrections

- `POST /v1/corrections` creates a correction rule.
- `GET /v1/corrections?enabled=true|false` lists rules.
- `GET /v1/corrections/{rule_id}` returns one rule.
- `PATCH /v1/corrections/{rule_id}` updates supplied fields.
- `DELETE /v1/corrections/{rule_id}` deletes a rule and returns `204`.

Create body:

```json
{
  "source_phrase": "Acme corp",
  "replacement_phrase": "ACME Corp",
  "context_terms": [],
  "tags": ["company"],
  "enabled": true,
  "priority": 0
}
```

## Confirmed-text feedback

`POST /v1/transcriptions/{id}/feedback` attaches rule feedback and/or approves
an exact raw-transcript override. At least one of `rule_ids` or
`confirmed_text` is required.

```json
{
  "rule_ids": [],
  "confirmed_text": "the final text confirmed by the user",
  "capture_method": "active_field_on_enter",
  "client_metadata": {
    "client": "CtrlSpeak",
    "version": "0.5.3"
  }
}
```

The desktop client binds feedback to the original transcription ID, API URL,
and in-memory credential so a later settings change cannot reroute it.

## OpenAPI

The running service publishes the canonical machine-readable contract at
`GET /openapi.json`, Swagger UI at `GET /docs`, and ReDoc at `GET /redoc`.
